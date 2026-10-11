# Runtime label independence check

This check exercises the unchanged deployment CLI and each fitted package with every role-data record removed from its runtime manifest. A Python audit hook is installed before deployment imports and refuses original role-data paths, label archives and embargoed archive patterns. The source manifest contains only dictionaries for pinned file records, so the recursive inventory covers the concrete admitted manifest.

The hook covers the actual pinned Python loaders and their file/hash operations. It is not an operating system sandbox against arbitrary native direct I/O, a malicious replacement checkpoint or unexamined code. The reported observation supports the current runtime's lack of a task-label or protected-label dependency. It is not a universal security guarantee.

Earlier comprehensive source admission and replay queue preflights hashed permitted role-label files. That infrastructure access remains documented. The runtime independence unit's own preflight deliberately contains only frozen runtime artifacts and registered campaign-private raw inputs, provider keys, private coins and expected wires; it includes no role-data or label archive pin.
