import { readFile, readdir } from "node:fs/promises";
import { basename, dirname, join, relative, resolve } from "node:path";
import { spawnSync } from "node:child_process";

const projectRoot = process.cwd();
const configPath = join(projectRoot, ".github", "marimo-publish.tsv");
const exportScriptPath = join(projectRoot, "scripts", "export_marimo.mjs");
const config = new Map();
const discoveredPaths = new Set();

for (const [lineIndex, line] of (await readFile(configPath, "utf8")).split(/\r?\n/).entries()) {
  const trimmed = line.trim();
  if (!trimmed || trimmed.startsWith("#")) continue;

  const [notebookPath, variants, ...extraFields] = trimmed.split(/\s+/);
  if (!notebookPath || !variants || extraFields.length > 0) {
    throw new Error(`${configPath}:${lineIndex + 1} must contain a notebook path and comma-separated variants`);
  }
  if (config.has(notebookPath)) {
    throw new Error(`${configPath}:${lineIndex + 1} repeats notebook path ${notebookPath}`);
  }
  config.set(notebookPath, variants);
}

for (const root of ["blog", "talks"]) {
  for (const notebookPath of await findMarimoApps(join(projectRoot, root))) {
    const repositoryPath = relative(projectRoot, notebookPath);
    discoveredPaths.add(repositoryPath);
    const variants = (config.get(repositoryPath) ?? "read,slide,edit").split(",");
    const stem = basename(notebookPath, ".py");

    for (const variant of variants) {
      if (!["read", "slide", "edit"].includes(variant)) {
        throw new Error(
          `${configPath} configures unsupported export '${variant}' for ${repositoryPath}; use read, slide, or edit`,
        );
      }

      const outputPath = join(dirname(notebookPath), `${stem}_${variant}.html`);
      console.log(`Exporting ${repositoryPath} as ${variant} to ${relative(projectRoot, outputPath)}`);
      const result = spawnSync(
        process.execPath,
        [exportScriptPath, notebookPath, variant, outputPath],
        { cwd: projectRoot, stdio: "inherit" },
      );
      if (result.error) throw result.error;
      if (result.status !== 0) {
        throw new Error(`Could not export ${repositoryPath} as ${variant}`);
      }
    }
  }
}

for (const configuredPath of config.keys()) {
  const notebookPath = resolve(projectRoot, configuredPath);
  if (!notebookPath.startsWith(`${projectRoot}/`)) {
    throw new Error(`${configPath} contains a path outside the project: ${configuredPath}`);
  }
  if (!discoveredPaths.has(relative(projectRoot, notebookPath))) {
    throw new Error(`${configPath} contains invalid notebook path: ${configuredPath}`);
  }
}

async function findMarimoApps(directory) {
  const found = [];
  let entries;
  try {
    entries = await readdir(directory, { withFileTypes: true });
  } catch (error) {
    if (error.code === "ENOENT") return found;
    throw error;
  }

  for (const entry of entries) {
    const entryPath = join(directory, entry.name);
    if (entry.isDirectory()) {
      if (entry.name.startsWith(".")) continue;
      found.push(...(await findMarimoApps(entryPath)));
    } else if (entry.isFile() && entry.name.endsWith(".py") && !entry.name.startsWith(".")) {
      const source = await readFile(entryPath, "utf8");
      if (/^[\t ]*[A-Za-z_]\w*[\t ]*=[\t ]*(?:marimo|mo)\.App\(/m.test(source)) {
        found.push(entryPath);
      }
    }
  }

  return found;
}
