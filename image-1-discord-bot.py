# ───── image_prep utilities ──────────────────────────────────────────────────
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import uuid

from PIL import Image
import torch
import numpy as np
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from io import BytesIO
import uuid
from PIL import Image
import base64
import asyncio
import openai
import logging
import discord
from discord.ui import View
from pathlib import Path
import re                                    # ← add at top of file if missing
from PIL import ImageFilter                  # ← already installed with Pillow

# where masks and converted images go
SAVE_DIR = Path("generated")
SAVE_DIR.mkdir(exist_ok=True)

# one-time CLIPSeg globals (module scope)
_SEG_PROCESSOR = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
_SEG_MODEL     = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

def segment_mask(image_path: Path, text: str, threshold: float = 0.3) -> Path:
    """
    Zero-shot mask for “text” on image_path via CLIPSeg.
    White = pixels to modify, black = pixels to preserve.
    """
    im = Image.open(image_path).convert("RGB")

    # 1️⃣ tokenize the text
    text_inputs = _SEG_PROCESSOR.tokenizer(
        [text], padding=True, truncation=True, return_tensors="pt"
    )

    # 2️⃣ preprocess the image (we only need the pixel_values array)
    image_batch = _SEG_PROCESSOR.image_processor(images=[im])["pixel_values"]

    # 3️⃣ coerce into a plain array, then into a torch.Tensor
    arr = np.asarray(image_batch)
    pixel_values = torch.tensor(arr)

    # 4️⃣ run the model
    with torch.no_grad():
        outputs = _SEG_MODEL(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            pixel_values=pixel_values,
        )

    # 5️⃣ threshold & upsample
    logits = outputs.logits[0, 0].cpu().numpy()
    mask_arr = (logits > threshold).astype(np.uint8) * 255
    mask = Image.fromarray(mask_arr, mode="L").resize(im.size, Image.LANCZOS)

    # 6️⃣ save & return
    dst = SAVE_DIR / f"{uuid.uuid4()}_auto_mask.png"
    mask.save(dst, "PNG")
    return dst

class ImagePrep:
    @staticmethod
    def to_png(src: Path) -> Path:
        if src.suffix.lower() == ".png":
            return src
        dst = SAVE_DIR / f"{uuid.uuid4()}.png"
        Image.open(src).save(dst, "PNG")
        return dst

    @staticmethod
    def to_square(src: Path, side: int = 1024, bg_color: Tuple[int,int,int]=(255,255,255)) -> Path:
        im = Image.open(src).convert("RGBA")
        im.thumbnail((side, side), Image.LANCZOS)
        canvas = Image.new("RGB", (side, side), bg_color)
        canvas.paste(im, ((side - im.width)//2, (side - im.height)//2), im)
        dst = SAVE_DIR / f"{uuid.uuid4()}.png"
        canvas.save(dst, "PNG")
        return dst

    @staticmethod
    def white_mask(for_image: Path) -> Path:
        w, h = Image.open(for_image).size
        mask = Image.new("L", (w, h), 255)
        dst = SAVE_DIR / f"{for_image.stem}_mask.png"
        mask.save(dst, "PNG")
        return dst

    @staticmethod
    def prep_for_edit(src: Path) -> Tuple[Path, Path]:
        png = ImagePrep.to_png(src)
        sqr = ImagePrep.to_square(png, 1024)
        mask = ImagePrep.white_mask(sqr)
        return sqr, mask

    @staticmethod
    def prep_for_variation(src: Path) -> Path:
        png = ImagePrep.to_png(src)
        return ImagePrep.to_square(png, 1024)

    @staticmethod
    def prep_for_outpaint(src: Path) -> Tuple[Path, Path]:
        png = ImagePrep.to_png(src)
        sqr = ImagePrep.to_square(png, 1024)
        im = Image.open(png).convert("RGBA")
        im.thumbnail((1024,1024), Image.LANCZOS)
        x = (1024 - im.width)//2
        y = (1024 - im.height)//2
        w, h = im.width, im.height
        border_mask = Image.new("L", (1024,1024), 0)
        border_mask.paste(255, (0, 0, 1024, y))
        border_mask.paste(255, (0, y+h, 1024, 1024))
        border_mask.paste(255, (0, y, x, y+h))
        border_mask.paste(255, (x+w, y, 1024, y+h))
        mpath = SAVE_DIR / f"{uuid.uuid4()}_border_mask.png"
        border_mask.save(mpath, "PNG")
        return sqr, mpath

# ───── imports ───────────────────────────────────────────────────────────────
import os, re, shlex, asyncio, base64, sqlite3, logging, sys
from typing import Tuple

import aiohttp, discord, openai
from discord.ext import commands
from discord.ui import View, Modal, TextInput

# ───── logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)5s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("img-bot")

# ───── configuration ────────────────────────────────────────────────────────
openai.api_key = "YOUR_API_KEY"
BOT_PREFIX       = "!img"
VISION_MODEL     = "gpt-4o-mini"
SAVE_DIR         = Path("generated"); SAVE_DIR.mkdir(exist_ok=True)
SHARE_CHANNEL_ID = YOUR_CHANNEL_ID

CHOICES = {
    "size":   {"1024x1024","1024x1536","1536x1024"},
    "style":  {"vivid","natural","anime","cinematic","pixel"},
    "quality":{"low","medium","high"},
    "format": {"png","jpeg","webp"},
}
FLAGS = set(CHOICES) | {"n","transparent","seed"}
EXTRA_STYLE = {
    "anime":     "anime style, vibrant cel shading, clean line art, ",
    "cinematic": "cinematic lighting, ultra-wide shot, 35 mm film grain, ",
    "pixel":     "retro 16-bit pixel-art, crisp sprites, no anti-aliasing, ",
}
FLAG_RE = re.compile(r"--(\w+)(?:=(\S+))?")

# ───── persistence (SQLite) ─────────────────────────────────────────────────
DB = sqlite3.connect("gen_cache.db")
DB.execute("""CREATE TABLE IF NOT EXISTS gen(
  id      INTEGER PRIMARY KEY,
  user    INTEGER,
  prompt  TEXT,
  file    TEXT,
  size    TEXT,
  quality TEXT,
  seed    INTEGER
)""")
DB.commit()

class Gen:
    def __init__(self, prompt, file, size, qual, seed):
        self.prompt, self.file, self.size, self.qual, self.seed = (
            prompt, Path(file), size, qual, seed
        )

def cache_save(mid, uid, g):
    DB.execute("REPLACE INTO gen VALUES (?,?,?,?,?,?,?)",
               (mid, uid, g.prompt, str(g.file), g.size, g.qual, g.seed))
    DB.commit()

def cache_load(mid) -> Optional[Gen]:
    row = DB.execute(
        "SELECT prompt,file,size,quality,seed FROM gen WHERE id=?", (mid,)
    ).fetchone()
    return Gen(*row) if row else None

# ───── helpers ──────────────────────────────────────────────────────────────
async def download_url(url, dest):
    url = str(url)
    async with aiohttp.ClientSession() as s:
        async with s.get(url) as r:
            r.raise_for_status()
            dest.write_bytes(await r.read())

async def download_att(att: discord.Attachment) -> Path:
  """
  Download a Discord attachment into our SAVE_DIR, giving it a short
  UUID-based filename (preserving its original extension).
  """
  # grab the extension (fallback to .png if none)
  ext = Path(att.filename).suffix or ".png"
  dst = SAVE_DIR / f"{uuid.uuid4()}{ext}"
  await att.save(dst)
  return dst

async def caption(url: str) -> str:
    log.info("caption  | %s", url)
    msg = [{
        "role":"user","content":[
            {"type":"image_url","image_url":{"url":url}},
            {"type":"text","text":"Describe this image in one sentence."}
        ]
    }]
    r = await asyncio.to_thread(
        openai.chat.completions.create,
        model=VISION_MODEL, messages=msg
    )
    return r.choices[0].message.content.strip()

def parse(tokens: List[str]) -> Tuple[str, Dict[str,str]]:
    prompt_parts: List[str] = []
    flags: Dict[str,str]   = {}
    it = iter(tokens)
    for tok in it:
        if tok.startswith("--"):
            key, eq, tail = tok.lstrip("-").partition("=")
            if eq:
                val = tail
            else:
                nxt = next(it, None)
                if nxt is None or nxt.startswith("--"):
                    raise ValueError(f"--{key} needs a value")
                val = nxt
            flags[key] = val
        else:
            prompt_parts.append(tok)
    return " ".join(prompt_parts).strip(), flags

def price(quality: str) -> float:
    return {"low":0.02,"medium":0.04,"high":0.08}.get(quality,0.04)

# ───── tighten hidden instruction ─────────────────────────────────────────────
HIDDEN_INSTRUCTION = (
    "You are a precise image inpainting assistant. "
    "Only recolor or retouch existing pixels; do not introduce, remove, or redraw any "
    "objects, symbols, text, or shapes. "
    "Always preserve composition, lighting, perspective, and textures exactly as they are. "
    "Only apply the exact modification described below: "
)

# …later, replace variate() with:
async def variate(g: Gen, uid: str) -> Gen:
    """
    Produce a faithful “variation” by doing a full‐canvas masked edit
    (never falling back to create_variation).
    """
    log.info("variate   | user=%s | file=%s", uid, g.file)

    src_png, mask_png = ImagePrep.prep_for_edit(g.file)

    full_prompt = (
        HIDDEN_INSTRUCTION
        + "Generate a subtle variation of the original image, changing as little as possible."
    )

    try:
        r = await asyncio.to_thread(
            openai.images.edit,
            model="gpt-image-1",
            image=open(src_png,  "rb"),
            mask=open(mask_png, "rb"),
            prompt=full_prompt,
            n=1,
            size="1024x1024",
            user=uid,
        )
        d = r.data[0]
        out = SAVE_DIR / f"{uuid.uuid4()}.png"
        if getattr(d, "url", None):
            await download_url(str(d.url), out)
        else:
            out.write_bytes(base64.b64decode(d.b64_json))

        return Gen(f"{g.prompt} (variation)", out, "1024x1024", g.qual, g.seed)

    except openai.OpenAIError as e:
        log.warning("variate edit failed (%s) – falling back to regular generate", e)
        g2, _ = await generate(g.prompt, {"size": "1024x1024", "quality": g.qual}, uid)
        return Gen(f"{g.prompt} (variation)", g2.file, "1024x1024", g.qual, g2.seed)
        
# ───── scope classification ──────────────────────────────────────────────────
        
# operations that imply you really want to recolor/restore/enhance the whole image
GLOBAL_HINTS = {
    "enhance", "sharpen", "stylize", "denoise", "upscale",
}

# operations that imply you only want to touch a region or the subject itself
LOCAL_HINTS = {
    "colorize", "restore", "recolor", "repaint", "tint",
    "remove", "erase", "replace", "swap", "turn into", "add", "overlay",
    "background", "foreground", "sky", "face", "hair", "eyes",
    "shirt", "hat", "dog", "cat", "tree", "car", "window",
}

# ───── LLM‐assisted scope classifier ─────────────────────────────────────────

async def classify_scope(prompt: str, uid: str) -> str:
  """
  Ask the LLM if this edit is 'local' (a region) or 'global' (the whole image).
  Returns exactly "local" or "global".
  """
  messages = [
      {
          "role": "system",
          "content": (
              "You are an image-edit classification assistant.  "
              "Given a user’s edit prompt, decide if the instruction "
              "targets only a specific part of the image (reply 'local') "
              "or the entire image (reply 'global').  "
              "Respond with exactly one word: either 'local' or 'global'."
          ),
      },
      {"role": "user", "content": f"Edit prompt: {prompt}"}
  ]
  
  comp = await asyncio.to_thread(
      openai.chat.completions.create,
      model=VISION_MODEL,
      messages=messages,
      user=uid,
  )
  answer = comp.choices[0].message.content.strip().lower()
  return "local" if answer.startswith("local") else "global"

# ★ robust, image-guided remix helper ───────────────────────────────────
async def edit_img(g: Gen, user_prompt: str, uid: str) -> Gen:
    """
    1. Optional square out-paint (best-effort).
    2. LOCAL vs GLOBAL mask decision (keywords → LLM fallback).
    3. Dilate LOCAL mask 8 px so pin-stripes / halos can’t slip through.
    4. One images.edit call (PNG-header retry).  No silent fallback.
    """
    import numpy as np
    from PIL import Image, ImageFilter

    base_prompt = g.prompt or user_prompt

    def make_full_prompt(scope: str) -> str:
        bg_guard = (" Keep every pixel outside the masked region untouched."
                    if scope == "local" else "")
        return (HIDDEN_INSTRUCTION + user_prompt +
                bg_guard + " Do not add new objects, symbols, or text.")

    log.info("edit_img | uid=%s | file=%s", uid, g.file)

    # 1️⃣   best-effort border out-paint to 1024² ------------------------------
    try:
        sq_img, border_mask = ImagePrep.prep_for_edit(g.file)
        r0 = await asyncio.to_thread(
            openai.images.edit,
            model="gpt-image-1",
            image=open(sq_img,  "rb"),
            mask=open(border_mask, "rb"),
            prompt=base_prompt,
            n=1, size="1024x1024", user=uid,
        )
        tmp = SAVE_DIR / f"{uuid.uuid4()}.png"
        d0  = r0.data[0]
        if getattr(d0, "url", None):
            await download_url(d0.url, tmp)
        else:
            tmp.write_bytes(base64.b64decode(d0.b64_json))
        g = Gen(base_prompt, tmp, "1024x1024", g.qual, g.seed)
    except openai.OpenAIError as e:
        log.warning("out-paint skipped (%s)", e)

    # 2️⃣   scope → mask -------------------------------------------------------
    def keyword_scope(p: str) -> Optional[str]:
        p = p.lower()
        if any(k in p for k in LOCAL_HINTS):  return "local"
        if any(k in p for k in GLOBAL_HINTS): return "global"
        return None

    scope = keyword_scope(user_prompt) or await classify_scope(user_prompt, uid)

    src_png, full_mask = ImagePrep.prep_for_edit(g.file)
    if scope == "local":
        # naive noun-phrase → before “to/with/in…”
        obj_phrase = re.split(r"\b(to|with|in|on|into|onto)\b",
                              user_prompt, 1)[0].strip()
        mask_png   = segment_mask(src_png, obj_phrase, threshold=0.35)

        # —— 2️⃣.a  dilate 8 px to close pin-stripes ——
        m = Image.open(mask_png).convert("L").filter(ImageFilter.MaxFilter(9))
        m.save(mask_png)
    else:
        mask_png = full_mask

    full_prompt = make_full_prompt(scope)

    # 3️⃣   one images.edit call (header-retry) -------------------------------
    async def run_edit(mask_path: Path):
        return await asyncio.to_thread(
            openai.images.edit,
            model="gpt-image-1",
            image=open(src_png, "rb"),
            mask=open(mask_path, "rb"),
            prompt=full_prompt,
            n=1, size="1024x1024", user=uid,
        )

    try:
        r1 = await run_edit(mask_png)
    except openai.BadRequestError as e:
        if "invalid_mask_image_format" in str(e).lower():
            log.warning("mask header glitch – one re-encode retry")
            fixed = SAVE_DIR / f"{uuid.uuid4()}_mask_fix.png"
            Image.open(mask_png).save(fixed, "PNG", optimize=False)
            r1 = await run_edit(fixed)
        else:
            raise

    # 4️⃣   save & return ------------------------------------------------------
    d1  = r1.data[0]
    out = SAVE_DIR / f"{uuid.uuid4()}.png"
    if getattr(d1, "url", None):
        await download_url(d1.url, out)
    else:
        out.write_bytes(base64.b64decode(d1.b64_json))

    return Gen(user_prompt, out, "1024x1024", g.qual, g.seed)

# ───── Discord UI & bot setup ────────────────────────────────────────────────
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix=lambda _b,m:BOT_PREFIX, intents=intents)

# ───── HelpView (paged help embed) ──────────────────────────────────────────
class HelpView(View):
    """Three-page embed with Prev / Next / Close."""
    def __init__(self):
        super().__init__(timeout=300)
        self.pages = [
            discord.Embed(title="GPT-Image-1 Bot", description="Generate · Remix · Upscale · Share"),
            discord.Embed(
                title="Flags & defaults", description=(
                    "**Defaults:** png 1024²  medium  vivid\n"
                    "`--size` 1024² / 1024×1536 / 1536×1024\n"
                    "`--quality` low / medium / high\n"
                    "`--style` vivid / natural / anime / cinematic / pixel\n"
                    "`--format` png / jpeg / webp\n"
                    "`--transparent`, `--seed 42`, `--n 3`"
                )
            ),
            discord.Embed(
                title="Examples", description=(
                    "`!img neon koi --style cinematic --size 1536x1024`\n"
                    "mention + image → caption & buttons\n"
                    "reply *laser eyes* to add lasers"
                )
            ),
        ]
        self.i = 0; self._sync()

    def _sync(self):
        self.prev.disabled = (self.i == 0)
        self.next.disabled = (self.i == len(self.pages)-1)

    @discord.ui.button(label="⏪ Prev", style=discord.ButtonStyle.secondary)
    async def prev(self, inter, _):
        self.i -= 1; self._sync()
        await inter.response.edit_message(embed=self.pages[self.i], view=self)

    @discord.ui.button(label="Next ⏩", style=discord.ButtonStyle.secondary)
    async def next(self, inter, _):
        self.i += 1; self._sync()
        await inter.response.edit_message(embed=self.pages[self.i], view=self)

    @discord.ui.button(label="✖ Close", style=discord.ButtonStyle.danger)
    async def close(self, inter, _):
        await inter.message.delete(); self.stop()

# ───── Remix modal (uses edit_img) ──────────────────────────────────────────
class RemixModal(Modal):
    def __init__(self, g):
        super().__init__(title="Remix")
        self.g = g
        self.t = TextInput(label="New prompt")
        self.add_item(self.t)

    async def on_submit(self, inter):
        await inter.response.defer()
        async with inter.channel.typing():
            g2 = await edit_img(self.g, self.t.value, str(inter.user.id))
        m = await inter.followup.send(
            content=f"Prompt: **{g2.prompt}**",
            file=discord.File(g2.file),
            view=ImgView(g2, inter.user.id)
        )
        cache_save(m.id, inter.user.id, g2)

async def generate(prompt: str, flags: Dict[str,str], uid: str) -> tuple[Gen, float]:
    """
    Fresh text→image gen honoring:
      --size, --quality, --style, --format, --transparent, --seed, --n
    Returns (Gen, cost).
    """
    # pull flags (with your defaults)
    size       = flags.get("size", "1024x1024")
    quality    = flags.get("quality", "medium")
    n          = int(flags.get("n", 1))
    style      = flags.get("style")
    img_format = flags.get("format", "png")
    transparent= "transparent" in flags
    seed       = flags.get("seed")
    
    # style prefix
    if style in EXTRA_STYLE:
        prompt = EXTRA_STYLE[style] + prompt
    
    # images.generate call
    params = {
        "model": "gpt-image-1",
        "prompt": prompt,
        "n": n,
        "size": size,
        "user": uid,
    }
    if seed is not None:
        params["seed"] = int(seed)
    
    resp = await asyncio.to_thread(openai.images.generate, **params)
    
    # save each result (we'll just take the first)
    out_path = SAVE_DIR / f"{uuid.uuid4()}.{img_format}"
    data = resp.data[0]
    if getattr(data, "url", None):
        await download_url(str(data.url), out_path)
    else:
        out_path.write_bytes(base64.b64decode(data.b64_json))
    
    # cost calc remains quality‐based
    cost = price(quality) * n
    
    # return Gen (show only prompt+style) and cost
    return Gen(prompt, out_path, size, quality, getattr(data, "seed", None)), cost

log = logging.getLogger("img-bot")
    
# ───── Image view buttons ───────────────────────────────────────────────────
class ImgView(View):
  def __init__(self, g: Gen, uid: int):
      super().__init__(timeout=None)
      self.g, self.uid = g, uid

  async def interaction_check(self, inter):
      if inter.user.id != self.uid:
          await inter.response.send_message(
              "Only the creator can use these buttons.", ephemeral=True
          )
          return False
      return True

  # ─── 🔍 Upscale – true 16:9 out-paint to 1536 × 1024 ────────────────────────
  @discord.ui.button(label="🔍 Upscale", style=discord.ButtonStyle.primary)
  async def up(self, inter, _):
      await inter.response.defer(ephemeral=True)
  
      # legal landscape canvas
      TARGET_SIZE = "1536x1024"
      W, H        = (1536, 1024)
      log.info("Upscale canvas target: %s", TARGET_SIZE)
  
      # 1️⃣ build 1536×1024 RGB canvas with the source centred
      src = Image.open(self.g.file).convert("RGBA")
      src.thumbnail((W, H), Image.LANCZOS)
  
      canvas = Image.new("RGB", (W, H), (0, 0, 0))
      x0, y0 = (W - src.width) // 2, (H - src.height) // 2
      canvas.paste(src.convert("RGB"), (x0, y0), src)
  
      # 2️⃣ RGBA mask: *transparent border = editable*, opaque centre = keep
      mask = Image.new("RGBA", (W, H), (0, 0, 0, 0))        # fully-editable
      keep = Image.new("RGBA", (src.width, src.height), (0, 0, 0, 255))  # alpha 255
      mask.paste(keep, (x0, y0))                            # protect the middle
  
      # 3️⃣ freeze both to disk so Pillow can’t silently optimise dimensions
      img_path  = SAVE_DIR / f"{uuid.uuid4()}_img.png"
      mask_path = SAVE_DIR / f"{uuid.uuid4()}_mask.png"
      canvas.save(img_path, "PNG", optimize=False)
      mask.save(mask_path,   "PNG", optimize=False)
  
      log.info("PNG written: %s  %s", img_path.name, mask_path.name)
  
      hidden_prompt = (
          HIDDEN_INSTRUCTION +
          "Extend the picture seamlessly into the transparent border. "
          "Do not modify the opaque centre."
      )
  
      async def call_edit():
          return await asyncio.to_thread(
              openai.images.edit,
              model="gpt-image-1",
              image=open(img_path,  "rb"),
              mask=open(mask_path,  "rb"),
              prompt=hidden_prompt,
              n=1,
              size=TARGET_SIZE,
              user=str(self.uid),
          )
  
      # 4️⃣ single retry if the server still complains about the mask header
      try:
          resp = await call_edit()
      except openai.BadRequestError as e:
          if "invalid_mask_image_format" in str(e).lower():
              log.warning("mask header glitch – re-encoding once and retrying")
              fixed = SAVE_DIR / f"{uuid.uuid4()}_mask_fix.png"
              Image.open(mask_path).save(fixed, "PNG", optimize=False)
              mask_path = fixed
              resp = await call_edit()
          else:
              raise
  
      # 5️⃣ save & send the result
      out = SAVE_DIR / f"{uuid.uuid4()}.png"
      d   = resp.data[0]
      if getattr(d, "url", None):
          await download_url(str(d.url), out)
      else:
          out.write_bytes(base64.b64decode(d.b64_json))
  
      g2 = Gen(self.g.prompt or "(upscaled)", out, TARGET_SIZE, "high", self.g.seed)
      m  = await inter.followup.send(
          content=f"Prompt: **{g2.prompt}**",
          file=discord.File(g2.file),
          view=ImgView(g2, self.uid),
      )
      cache_save(m.id, self.uid, g2)
      
  # ─── other buttons unchanged ────────────────────────────────────────────
  @discord.ui.button(label="🎨 Remix", style=discord.ButtonStyle.secondary)
  async def rm(self, inter, _):
      await inter.response.send_modal(RemixModal(self.g))

  @discord.ui.button(label="🔁 Variate", style=discord.ButtonStyle.secondary)
  async def var(self, inter, _):
      await inter.response.send_message("🔄 Variation…", ephemeral=True)
      async with inter.channel.typing():
          g2 = await variate(self.g, str(self.uid))
      m = await inter.followup.send(
          content=f"Prompt: **{g2.prompt}**",
          file=discord.File(g2.file),
          view=ImgView(g2, self.uid),
      )
      cache_save(m.id, self.uid, g2)

  @discord.ui.button(label="📤 Share", style=discord.ButtonStyle.success)
  async def sh(self, inter, _):
      if not SHARE_CHANNEL_ID:
          await inter.response.send_message("Share channel not configured.", ephemeral=True)
          return
      channel = bot.get_channel(SHARE_CHANNEL_ID)
      await channel.send(
          content=f"Prompt: **{self.g.prompt}**\nShared by {inter.user.mention}",
          file=discord.File(self.g.file),
      )
      await inter.response.send_message("✅ Shared!", ephemeral=True)

# ───── events ───────────────────────────────────────────────────────────────
@bot.event
async def on_ready():
    log.info("Logged in as %s", bot.user)

# ───── events ───────────────────────────────────────────────────────────────
@bot.event
async def on_message(msg: discord.Message):
    if msg.author.bot:
        return

    mention = f"<@{bot.user.id}>"
    txt      = msg.content
    if txt.startswith(mention):
        txt = txt[len(mention):].lstrip()
    elif txt.startswith(BOT_PREFIX):
        txt = txt[len(BOT_PREFIX):].lstrip()
    else:
        txt = None

    # ─── help ----------------------------------------------------------------
    if msg.content.lower().strip() in {f"{BOT_PREFIX} help", f"{mention} help"}:
        await msg.reply(embed=HelpView().pages[0], view=HelpView())
        return

    # ─── mention + attachment  (ping-image workflow) -------------------------
    if msg.attachments and bot.user in msg.mentions:
        att         = msg.attachments[0]
        user_prompt = (txt or "").strip()

        async with msg.channel.typing():
            raw_path = await download_att(att)

            # ① user supplied prompt  →  immediate edit
            #    normalise the upload to a *clean* 1 024 × 1 024 PNG first
            if user_prompt:
                square_png, _ = ImagePrep.prep_for_edit(raw_path)
                g0 = Gen("(original)", square_png, "1024x1024", "medium", None)
                g1 = await edit_img(g0, user_prompt, str(msg.author.id))

                m = await msg.channel.send(
                    content=f"Prompt: **{g1.prompt}**",
                    file=discord.File(g1.file),
                    view=ImgView(g1, msg.author.id),
                )
                cache_save(m.id, msg.author.id, g1)
                return

            # ② no prompt  →  caption + buttons
            caption_txt = await caption(att.url)

        g = Gen("", raw_path, "1024x1024", "medium", None)
        m = await msg.channel.send(
            content=caption_txt,
            file=discord.File(raw_path),
            view=ImgView(g, msg.author.id),
        )
        cache_save(m.id, msg.author.id, g)
        return

    # ─── blank ping → help ---------------------------------------------------
    if txt is not None and not txt.strip():
        await msg.reply(embed=HelpView().pages[0], view=HelpView())
        return

    # ─── plain text prompt → fresh generate ----------------------------------
    if txt is not None:
        prompt, flags = parse(shlex.split(txt))
        if not prompt and not flags:
            await msg.reply(embed=HelpView().pages[0], view=HelpView())
            return
        try:
            async with msg.channel.typing():
                g, _ = await generate(prompt, flags, str(msg.author.id))
        except openai.OpenAIError as e:
            if getattr(e, "code", None) == "moderation_blocked":
                await msg.reply(
                    "⚠️ Your request was blocked by the safety filter. "
                    "Please try re-phrasing your prompt."
                )
                return
            raise
        m = await msg.reply(
            content=f"Prompt: **{g.prompt}**",
            file=discord.File(g.file),
            view=ImgView(g, msg.author.id),
        )
        cache_save(m.id, msg.author.id, g)
        return

    # ─── reply-remix ---------------------------------------------------------
    if msg.reference and msg.content.strip():
        ref = (
            msg.reference.resolved
            or await msg.channel.fetch_message(msg.reference.message_id)
        )
        if ref and ref.author.id == bot.user.id:
            g0 = cache_load(ref.id)
            if g0:
                try:
                    async with msg.channel.typing():
                        g1 = await edit_img(g0, msg.content.strip(), str(msg.author.id))
                except openai.OpenAIError as e:
                    if getattr(e, "code", None) == "moderation_blocked":
                        await msg.reply(
                            "⚠️ Your edit request was blocked by the safety filter. "
                            "Try a different instruction."
                        )
                        return
                    raise
                m = await msg.reply(
                    content=f"Prompt: **{g1.prompt}**",
                    file=discord.File(g1.file),
                    view=ImgView(g1, msg.author.id),
                )
                cache_save(m.id, msg.author.id, g1)
                return

    # ─── allow other commands ------------------------------------------------
    await bot.process_commands(msg)

# ───── run ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    token="TOKEN"
    bot.run(token)
