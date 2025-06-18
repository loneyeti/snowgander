export const fileTypeFromBuffer = jest
  .fn()
  .mockResolvedValue({ mime: "image/png", ext: "png" });
