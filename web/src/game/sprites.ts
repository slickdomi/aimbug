import { DATA_URL } from "../config";

export interface SpriteCell {
  x: number; // cell index in the atlas
  u0: number; v0: number; u1: number; v1: number; // sprite rect inside the cell (0..1)
  credit: string; license: string; licenseUrl: string; url: string;
}

export interface SpriteAtlas {
  texture: GPUTexture;
  sampler: GPUSampler;
  cells: SpriteCell[];
}

export const spriteUrl = DATA_URL.replace(/data\/malecns-v1$/, "sprites");

/** Atlas as a plain image, for the Canvas 2D fallback renderer. */
export async function loadSpriteImage(): Promise<{ image: ImageBitmap; cells: SpriteCell[] }> {
  const meta: { cells: SpriteCell[] } = await (await fetch(`${spriteUrl}/females.json`)).json();
  const image = await createImageBitmap(await (await fetch(`${spriteUrl}/females.png`)).blob());
  return { image, cells: meta.cells };
}

/** Loads the female-fly photo atlas as a mipmapped texture (mips built with canvas downscaling). */
export async function loadSprites(device: GPUDevice): Promise<SpriteAtlas> {
  const meta: { cellSize: number; cells: SpriteCell[] } = await (await fetch(`${spriteUrl}/females.json`)).json();
  const blob = await (await fetch(`${spriteUrl}/females.png`)).blob();
  const bitmap = await createImageBitmap(blob, { premultiplyAlpha: "none", colorSpaceConversion: "none" });
  const levels = Math.floor(Math.log2(Math.max(bitmap.width, bitmap.height))) + 1;
  const texture = device.createTexture({
    size: [bitmap.width, bitmap.height],
    format: "rgba8unorm",
    mipLevelCount: levels,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  let w = bitmap.width;
  let h = bitmap.height;
  for (let level = 0; level < levels; level++) {
    const canvas = new OffscreenCanvas(w, h);
    const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(bitmap, 0, 0, w, h);
    const pixels = ctx.getImageData(0, 0, w, h).data;
    device.queue.writeTexture({ texture, mipLevel: level }, pixels, { bytesPerRow: 4 * w }, [w, h]);
    w = Math.max(1, w >> 1);
    h = Math.max(1, h >> 1);
  }
  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear", mipmapFilter: "linear" });
  return { texture, sampler, cells: meta.cells };
}
