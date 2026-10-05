# Session cleanup and idle storage maintenance

`artemis trace purge SESSION_UUID --path /path/to/traces --json` removes one finished
session transactionally, including history chunks and image ownership. Normal purge
never runs VACUUM. Session indexes avoid scanning all other jobs' rows. Active local
sessions are rejected. A database failure is reported with SQLite code/name, DB/WAL
sizes, free space and page counts. A missing database is rejected without creating
one. If initialization or a liveness check fails, artifacts are preserved. After
inactivity is verified, storage may still reclaim that session's own files if a later
SQL deletion fails; caller-supplied compiled paths are retained on database failure.
Storage and CLI file deletion share a per-command report populated only after a
deletion succeeds. The summary includes those confirmed removals, including
successfully deleted last-owner screenshots and their bytes, with
`cleanup_status=partial` and exit code zero when anything was removed, even if the
SQL transaction failed. Files disappearing during another cleanup do not turn a
failed purge into success or contribute to its reclaimed-byte count.
Inspect `cleanup_status` and `errors`; exit code zero alone does not mean database
rows were deleted. Already-missing paths do not count as reclaimed artifacts.
If another cleanup removes a target during deletion, that target contributes no
bytes and cleanup continues with the remaining artifacts. Permission and other
filesystem failures are still reported.

Screenshots are retained until their last tracked session owner is removed. Legacy
images remain conservatively unmanaged until idle maintenance inspects references.
Cache hits restore missing files. The old `--prune-images-older-than` purge option
emits a warning in both plain-text output and the JSON `warnings` list instead of
removing shared files solely by age.

## Legacy database recovery on Mac/Linux

1. Drain all Artemis users of the database, including direct CLI/MCP runs, and stop
   new submissions through the normal scheduler. Keep a backup if retained traces
   must be recoverable. Do not remove an active database's WAL/SHM files.
2. Run `.venv/bin/python -m artemis.main trace maintenance --path /absolute/path/to/traces --json`.
   This checks the exclusive activity lease, active session PIDs, and open database
   handles, then removes orphan history chunks and old unreferenced images. History
   deletion is batched; legacy trace/history image references are preserved.
3. If disk space must be returned to the filesystem, add `--compact`. VACUUM is
   explicit, requires at least twice the database size in free space, and checks for
   checkpoint contention before starting. Without compaction, SQLite reuses freed
   pages for future writes.
4. Check the JSON result and disk space before resuming submissions. Never run
   maintenance concurrently with an older deployment that does not honor the lease;
   PID/open-handle checks supplement, but cannot replace, draining those clients.

Default image grace is six hours (`--image-min-age-s`). Maintenance requires POSIX
activity leases; normal session/image operations also work on Windows. A lease is
held for the DataEngine lifetime and per storage operation. Read-only operations
share a lock on the existing parent directory; maintenance holds that lock exclusively
as well, preventing new readers from entering during cleanup or compaction. This
conservatively coordinates all databases in the same directory without creating
files for read-only clients. Readers wait at most 30 seconds for that lease, then
return a timeout asking the caller to retry trace inspection after maintenance.
Maintenance still refuses immediately when another operation holds a conflicting
lease. Schema initialization for maintenance happens only after acquiring the
exclusive lease.

With the companion Wayfinder ENG-2219 change, dispatched Apollo jobs use a separate
workspace/database per run and delete the entire owned workspace after evidence
handoff. This maintenance command is for legacy or direct Artemis shared storage.
Roll out both repositories to hosts 29 and 100 after draining, then validate a small
batch on distinct devices before restoring normal concurrency. These changes do not
repair invalid provider credentials or prove the cause of a host-level disk I/O error.

UIAutomator clients release only their local proxy on disconnect. They do not own
the device-wide server or Poco instrumentation. An already-registered service is
reported as an explicit ownership conflict; recovery belongs to the scheduler or
operator that holds the device lease. This avoids disrupting another live client.
