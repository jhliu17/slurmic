# Code Packing

**Code packing** snapshots your source tree into the job's output folder *before* the
job is submitted, and runs the job from that frozen copy instead of your live working
directory. It is enabled per job through the SLURM configuration:

```python
import slurmic

slurm_config = slurmic.SlurmConfig(
    mode="slurm",
    job_name="example_job",
    partition="your_partition",
    pack_code=True,        # turn packing on
    code_root=".",         # the tree to snapshot
)
```

When `pack_code=True`, slurmic copies the selected files from `code_root` into
`<output_path>/code/` and sets that directory as the job's working directory.

## Why pack code?

On a cluster, a job often does not start the instant you submit it — it waits in the
queue, and a job array or a chain of dependent jobs may keep launching for hours or
days. Meanwhile you keep editing the same source files. Without packing, every job
reads whatever happens to be on disk at the moment it *starts*, so a late edit can
silently change the behavior of an experiment you launched earlier.

Packing solves this by giving each submission an immutable snapshot:

- **Reproducibility** — a queued or long-running job executes the exact code that
  existed at submission time, no matter how the working tree changes afterwards.
- **Safe iteration** — you can continue developing, refactoring, or starting new
  experiments while previous jobs are still pending or running.
- **Isolation between jobs** — sequential and dependent jobs each carry their own
  copy, so they never interfere through shared source files.
- **A record of what ran** — the packed `code/` folder lives next to the job's logs
  and outputs, documenting precisely what produced a given result.

## What gets packed: the selection priority

Which files are copied depends on whether `code_root` contains a `.gitignore` file.
The two controls you configure are:

- **`code_file_suffixes`** — file extensions treated as "code" (default: `.py`,
  `.sh`, `.yaml`, `.toml`).
- **`exclude_code_folders`** — folders that are *always* hard-excluded (default:
  `.git`, `wandb`, `outputs`, `datasets`).

### With a `.gitignore` present

`.gitignore` becomes the primary discovery mechanism, and `code_file_suffixes` acts as
an **add-back** whitelist. A file is packed when:

```text
pack(file) = (NOT gitignored)  OR  (extension in code_file_suffixes)
```

| In `code_file_suffixes`? | Ignored by `.gitignore`? | Packed? | Reason |
| --- | --- | --- | --- |
| yes | no  | ✅ | not ignored (discovery) |
| yes | yes | ✅ | add-back by `code_file_suffixes` |
| no  | no  | ✅ | not ignored (discovery) |
| no  | yes | ❌ | ignored and not added back |

In other words: everything your repository would track is packed, **plus** any code
file you explicitly whitelist even if it is gitignored. The `.git` directory itself is
always ignored.

### Without a `.gitignore`

There is nothing to discover from, so slurmic falls back to the original behavior and
`code_file_suffixes` is the **sole** include filter:

```text
pack(file) = (extension in code_file_suffixes)
```

| In `code_file_suffixes`? | Packed? |
| --- | --- |
| yes | ✅ |
| no  | ❌ |

### `exclude_code_folders` always wins

Regardless of the mode above, any file inside an `exclude_code_folders` directory is
**never** packed. This hard exclusion cannot be overridden by `.gitignore` discovery or
by the `code_file_suffixes` add-back — it is your final safety valve for keeping large
or irrelevant directories (datasets, run outputs, experiment trackers) out of the
snapshot.

The full priority, highest first:

1. **`exclude_code_folders`** — hard-excludes whole folders (always wins).
2. **`code_file_suffixes`** — adds back matching files (rescues gitignored code; sole
   filter when there is no `.gitignore`).
3. **`.gitignore`** — discovers everything not ignored (only when present).

## Tips

- Keep large artifacts (datasets, checkpoints, run outputs) in `.gitignore` and/or
  `exclude_code_folders` so they are not copied into every job folder.
- If a needed config or script is gitignored but small, add its extension to
  `code_file_suffixes` to have it packed via add-back.
- With no `.gitignore`, only files matching `code_file_suffixes` ship — add the
  extensions your job needs.
