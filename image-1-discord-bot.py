"""
discord_image_bot.py – GPT-Image-1 Discord bot
(logging, paged help, true image-guided Remix, safe Upscale)
"""

# ───── image_prep utilities ──────────────────────────────────────────────────
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import uuid
from PIL import Image

SAVE_DIR = Path("generated")
SAVE_DIR.mkdir(exist_ok=True)

class ImagePrep:
    @staticmethod
    def to_png(src: Path) -> Path:
        if src.suffix.lower() == ".png":
            return src
        dst = SAVE_DIR / f"{uuid.uuid4()}.png"
        Image.open(src).save(dst, "PNG")
        return dst

    @staticmethod
    def to_square(src: Path, side: int = 1024) -> Tuple[Path, Tuple[int,int,int,int]]:
        """
        Letter-box *src* to a centred side×side PNG.
        Returns (new_png_path, paste_box) where paste_box = (x,y,w,h)
        """
        im = Image.open(src).convert("RGBA")
        w,h = im.size
        im.thumbnail((side,side), Image.LANCZOS)
        canvas = Image.new("RGBA", (side,side), (0,0,0,0))
        x = (side - im.width)//2
        y = (side - im.height)//2
        canvas.paste(im,(x,y),im)
        dst = SAVE_DIR / f"{uuid.uuid4()}.png"
        canvas.save(dst, "PNG")
        return dst, (x,y,im.width,im.height)

    @staticmethod
    def white_mask(for_image: Path) -> Path:
        w,h = Image.open(for_image).size
        mask = Image.new("L",(w,h),255)
        dst = SAVE_DIR / f"{for_image.stem}_mask.png"
        mask.save(dst,"PNG")
        return dst

    @staticmethod
    def prep_for_edit(src: Path) -> Tuple[Path, Path]:
        png = ImagePrep.to_png(src)
        sqr, _ = ImagePrep.to_square(png,1024)
        mask = ImagePrep.white_mask(sqr)
        return sqr, mask

    @staticmethod
    def prep_for_variation(src: Path) -> Path:
        png = ImagePrep.to_png(src)
        sqr, _ = ImagePrep.to_square(png,1024)
        return sqr

    @staticmethod
    def prep_for_outpaint(src: Path) -> Tuple[Path, Path]:
        png = ImagePrep.to_png(src)
        sqr, (x,y,w,h) = ImagePrep.to_square(png,1024)
        mask = Image.new("L",(1024,1024),0)
        mask.paste(255, (0,0,1024,y))
        mask.paste(255, (0,y+h,1024,1024))
        mask.paste(255, (0,y, x, y+h))
        mask.paste(255, (x+w,y,1024,y+h))
        mpath = SAVE_DIR/f"{uuid.uuid4()}_border_mask.png"
        mask.save(mpath,"PNG")
        return sqr, mpath

# ───── imports ───────────────────────────────────────────────────────────────
import os, re, shlex, asyncio, base64, uuid, sqlite3, logging, sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import aiohttp, discord, openai
from discord.ext import commands
from discord.ui import View, Modal, TextInput

from PIL import Image

# ───── logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)5s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("img-bot")

# ───── configuration ────────────────────────────────────────────────────────
openai.api_key = "KEY HERE"
BOT_PREFIX       = "!img"
VISION_MODEL     = "gpt-4o-mini"
SAVE_DIR         = Path("generated"); SAVE_DIR.mkdir(exist_ok=True)
SHARE_CHANNEL_ID = YOUR_DISCORD_CHANNEL_ID

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
    dest = SAVE_DIR / f"{uuid.uuid4()}_{att.filename}"
    await att.save(dest)
    return dest

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
            val = tail if eq else next(it, None)
            flags[key] = val
        else:
            prompt_parts.append(tok)
    return " ".join(prompt_parts).strip(), flags

def price(quality: str) -> float:
    return {"low":0.02,"medium":0.04,"high":0.08}.get(quality,0.04)

# ───── OpenAI wrappers ──────────────────────────────────────────────────────
async def generate(prompt: str, flags: Dict[str,str], uid: str) -> Tuple[Gen,float]:
    size  = flags.get("size","1024x1024")
    qual  = flags.get("quality","medium")
    style = flags.get("style","vivid")
    fmt   = flags.get("format","png")
    n     = int(flags.get("n",1))
    seed  = int(flags["seed"]) if "seed" in flags else None

    if style in EXTRA_STYLE:
        prompt = EXTRA_STYLE[style] + prompt

    payload = {
        "model":  "gpt-image-1",
        "prompt": prompt,
        "n":       n,
        "size":    size,
        "quality": qual,
        "user":    uid
    }
    if seed:
        payload["seed"] = seed
    if "transparent" in flags or (fmt=="png" and flags.get("transparent")==""):
        payload["transparent_background"] = True

    log.info("generate  | user=%s | flags=%s", uid, flags)
    r = await asyncio.to_thread(openai.images.generate, **payload)
    d = r.data[0]

    dest = SAVE_DIR / f"{uuid.uuid4()}.{fmt}"
    if getattr(d,"url",None):
        await download_url(str(d.url), dest)
    else:
        dest.write_bytes(base64.b64decode(d.b64_json))

    return Gen(prompt,dest,size,qual,seed), price(qual)

async def variate(g: Gen, uid: str) -> Gen:
    new_prompt = f"{g.prompt} (variation)"
    log.info("variate   | user=%s | file=%s", uid, g.file)

    src = ImagePrep.prep_for_variation(g.file)
    with open(src, "rb") as fp:
        r = await asyncio.to_thread(
            openai.images.create_variation,
            image=fp, n=1, size="1024x1024", user=uid
        )

    dest = SAVE_DIR / f"{uuid.uuid4()}.jpg"
    await download_url(str(r.data[0].url), dest)
    return Gen(new_prompt,dest,"1024x1024",g.qual,g.seed)

# ★ NEW : true image‐guided remix helper with border outpaint ────────────────
async def edit_img(g: Gen, new_prompt: str, uid: str) -> Gen:
    base_prompt = g.prompt or new_prompt
    combined    = f"{base_prompt}, {new_prompt}" if g.prompt else new_prompt
    log.info("edit      | user=%s | file=%s", uid, g.file)

    # Phase 1: outpaint edges
    try:
        square_png, border_mask = ImagePrep.prep_for_edit(g.file)
        r0 = await asyncio.to_thread(
            openai.images.edit,
            model="gpt-image-1",
            image=open(square_png,  "rb"),
            mask=open(border_mask, "rb"),
            prompt=base_prompt,
            n=1, size="1024x1024", user=uid
        )
        data0 = r0.data[0]
        out0  = SAVE_DIR / f"{uuid.uuid4()}.png"
        if getattr(data0,"url",None):
            await download_url(str(data0.url), out0)
        else:
            out0.write_bytes(base64.b64decode(data0.b64_json))
        g = Gen(base_prompt, out0, "1024x1024", g.qual, g.seed)
    except openai.OpenAIError as e:
        log.warning("outpaint step failed (%s), continuing", e)

    # Phase 2: full-canvas edit
    try:
        src_png, mask_png = ImagePrep.prep_for_edit(g.file)
        r1 = await asyncio.to_thread(
            openai.images.edit,
            model="gpt-image-1",
            image=open(src_png,  "rb"),
            mask=open(mask_png, "rb"),
            prompt=combined,
            n=1, size="1024x1024", user=uid
        )
        data1 = r1.data[0]
        out1  = SAVE_DIR / f"{uuid.uuid4()}.png"
        if getattr(data1,"url",None):
            await download_url(str(data1.url), out1)
        else:
            out1.write_bytes(base64.b64decode(data1.b64_json))
        return Gen(combined, out1, "1024x1024", g.qual, g.seed)
    except openai.OpenAIError as e:
        log.warning("full edit failed (%s) – falling back to generate", e)

    # Final fallback
    g2, _ = await generate(combined, {"size":g.size,"quality":g.qual}, uid)
    return g2

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

# ───── Image view buttons ───────────────────────────────────────────────────
class ImgView(View):
    def __init__(self, g:Gen, uid:int):
        super().__init__(timeout=None)
        self.g, self.uid = g, uid

    async def interaction_check(self, inter):
        if inter.user.id != self.uid:
            await inter.response.send_message("Only the creator can use these.", ephemeral=True)
            return False
        return True

    @discord.ui.button(label="🔍 Upscale", style=discord.ButtonStyle.primary)
    async def up(self, inter, _):
        await inter.response.send_message("🔄 Upscaling…", ephemeral=True)
        best = {"1024x1024":"1536x1024","1024x1536":"1024x1536","1536x1024":"1536x1024"}[self.g.size]
        async with inter.channel.typing():
            g2,_ = await generate(self.g.prompt,{"quality":"high","size":best},str(self.uid))
        m = await inter.followup.send(
            content=f"Prompt: **{g2.prompt}**",
            file=discord.File(g2.file),
            view=ImgView(g2, self.uid)
        )
        cache_save(m.id, self.uid, g2)

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
            view=ImgView(g2, self.uid)
        )
        cache_save(m.id, self.uid, g2)

    @discord.ui.button(label="📤 Share", style=discord.ButtonStyle.success)
    async def sh(self, inter, _):
        if not SHARE_CHANNEL_ID:
            await inter.response.send_message("Share channel not set.", ephemeral=True)
            return
        ch = bot.get_channel(SHARE_CHANNEL_ID)
        share_content = f"Prompt: **{self.g.prompt}**\nShared by {inter.user.mention}"
        await ch.send(content=share_content, file=discord.File(self.g.file))
        await inter.response.send_message("✅ Shared!", ephemeral=True)

# ───── events ───────────────────────────────────────────────────────────────
@bot.event
async def on_ready():
    log.info("Logged in as %s", bot.user)

@bot.event
async def on_message(msg: discord.Message):
    if msg.author.bot:
        return

    mention = f"<@{bot.user.id}>"
    txt = msg.content
    if txt.startswith(mention):
        txt = txt[len(mention):].lstrip()
    elif txt.startswith(BOT_PREFIX):
        txt = txt[len(BOT_PREFIX):].lstrip()
    else:
        txt = None

    # help
    if msg.content.lower().strip() in {f"{BOT_PREFIX} help", f"{mention} help"}:
        await msg.reply(embed=HelpView().pages[0], view=HelpView())
        return

    # mention + attachment
    if msg.attachments and bot.user in msg.mentions:
        att  = msg.attachments[0]
        ptxt = (txt or "").strip()
        async with msg.channel.typing():
            path = await download_att(att)

            # immediate edit if prompt given
            if ptxt:
                g0 = Gen("(original)", path, "1024x1024", "medium", None)
                g1 = await edit_img(g0, ptxt, str(msg.author.id))
                m = await msg.channel.send(
                    content=f"Prompt: **{g1.prompt}**",
                    file=discord.File(g1.file),
                    view=ImgView(g1, msg.author.id)
                )
                cache_save(m.id, msg.author.id, g1)
                return

            # else caption + buttons
            cap = await caption(att.url)

        g = Gen(cap, path, "1024x1024", "medium", None)
        m = await msg.channel.send(
            content=cap,
            file=discord.File(path),
            view=ImgView(g, msg.author.id)
        )
        cache_save(m.id, msg.author.id, g)
        return

    # blank ping → help
    if txt is not None and not txt.strip():
        await msg.reply(embed=HelpView().pages[0], view=HelpView())
        return

    # text prompt → generate
    if txt is not None:
        prompt, flags = parse(shlex.split(txt))
        if not prompt and not flags:
            await msg.reply(embed=HelpView().pages[0], view=HelpView())
            return
        try:
            async with msg.channel.typing():
                g, cost = await generate(prompt, flags, str(msg.author.id))
        except openai.OpenAIError as e:
            if getattr(e, "code", None) == "moderation_blocked":
                await msg.reply(
                    "⚠️ Your request was blocked by the safety filter. Please try rephrasing your prompt."
                )
                return
            raise
        m = await msg.reply(
            content=f"Prompt: **{g.prompt}**",
            file=discord.File(g.file),
            view=ImgView(g, msg.author.id)
        )
        cache_save(m.id, msg.author.id, g)
        return

    # reply-remix
    if msg.reference and msg.content.strip():
        ref = msg.reference.resolved or await msg.channel.fetch_message(msg.reference.message_id)
        if ref and ref.author.id == bot.user.id:
            g0 = cache_load(ref.id)
            if g0:
                try:
                    async with msg.channel.typing():
                        g1 = await edit_img(g0, msg.content.strip(), str(msg.author.id))
                except openai.OpenAIError as e:
                    if getattr(e, "code", None) == "moderation_blocked":
                        await msg.reply(
                            "⚠️ Your edit request was blocked by the safety filter. Try a different instruction."
                        )
                        return
                    raise
                m = await msg.reply(
                    content=f"Prompt: **{g1.prompt}**",
                    file=discord.File(g1.file),
                    view=ImgView(g1, msg.author.id)
                )
                cache_save(m.id, msg.author.id, g1)
                return

    await bot.process_commands(msg)

# ───── run ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    token="YOUR_DISCORD_TOKEN"
    if not token:
        raise SystemExit("DISCORD_TOKEN env var missing")
    bot.run(token)
