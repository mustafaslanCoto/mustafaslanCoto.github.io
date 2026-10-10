import { copyFile, mkdir, readdir } from "node:fs/promises";
import { dirname, join, relative, resolve } from "node:path";

const projectRoot = process.cwd();
const outputDirectory = resolve(
  projectRoot,
  process.env.QUARTO_PROJECT_OUTPUT_DIR || "docs",
);
const generatedVariant = /_(?:read|slide|edit)\.html$/;
let copiedCount = 0;

for (const root of ["blog", "talks"]) {
  await copyGeneratedExports(join(projectRoot, root), root);
}

console.log(`Copied ${copiedCount} marimo export(s) into ${relative(projectRoot, outputDirectory)}`);

async function copyGeneratedExports(sourceDirectory, repositoryPath) {
  let entries;
  try {
    entries = await readdir(sourceDirectory, { withFileTypes: true });
  } catch (error) {
    if (error.code === "ENOENT") return;
    throw error;
  }

  for (const entry of entries) {
    if (entry.name.startsWith(".")) continue;

    const sourcePath = join(sourceDirectory, entry.name);
    const relativePath = join(repositoryPath, entry.name);
    if (entry.isDirectory()) {
      await copyGeneratedExports(sourcePath, relativePath);
    } else if (entry.isFile() && generatedVariant.test(entry.name)) {
      const targetPath = join(outputDirectory, relativePath);
      await mkdir(dirname(targetPath), { recursive: true });
      await copyFile(sourcePath, targetPath);
      copiedCount += 1;
      console.log(`Copied ${relativePath} to ${relative(projectRoot, targetPath)}`);
    }
  }
}
