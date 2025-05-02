import axios from "axios";
import { Buffer } from "buffer"; // Import Buffer

export function computeResponseCost(tokens: number, cost: number): number {
  const tokenCost = cost / 1000000;
  return tokens * tokenCost;
}

/**
 * Fetches an image from a URL and returns its MIME type and base64 data.
 * Uses dynamic import for 'file-type' to support ESM.
 * @param url The URL of the image to fetch.
 * @returns A promise resolving to an object with mimeType and base64Data, or null if an error occurs.
 */
export async function getImageDataFromUrl(
  url: string
): Promise<{ mimeType: string; base64Data: string } | null> {
  try {
    const response = await axios.get(url, {
      responseType: "arraybuffer",
    });
    const buffer = Buffer.from(response.data, "binary");
    // Dynamically import file-type
    const { fileTypeFromBuffer } = await import("file-type");
    const type = await fileTypeFromBuffer(buffer);

    if (!type || !type.mime.startsWith("image/")) {
      console.error(`Invalid or non-image MIME type detected: ${type?.mime}`);
      return null;
    }
    const base64Data = buffer.toString("base64");
    return { mimeType: type.mime, base64Data };
  } catch (error) {
    console.error(`Error fetching or processing image from URL ${url}:`, error);
    return null;
  }
}
