# Local submodule patches

These patches record local compatibility changes for third-party submodules without
requiring write access to the upstream submodule repositories.

Apply from the repository root:

```bash
git -C thirdparty/DROID-SLAM apply ../../patches/droid-slam-cu12.patch
git -C thirdparty/GroundingDINO apply ../../patches/groundingdino-cu12.patch
```

Current DROID-SLAM nested submodule checkouts observed locally:

```text
thirdparty/DROID-SLAM/thirdparty/eigen    2f3c27c23a664bfed66d1d1a976de160fac47ae1
thirdparty/DROID-SLAM/thirdparty/lietorch e7df86554156b36846008d8ddbcc4d8521a16554
```
