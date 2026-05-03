# Moore submodule patches

These patches record the local dirty submodule state from the `moore` branch so
the Moore/MUSA adaptation can be restored without pushing to the upstream
third-party repositories.

Apply from the WorldScore repository root:

```bash
git -C thirdparty/DROID-SLAM apply ../../patches/moore/droid-slam-moore.patch
git -C thirdparty/DROID-SLAM/thirdparty/lietorch apply ../../../../patches/moore/droid-slam-lietorch-moore.patch
git -C thirdparty/Grounded-Segment-Anything apply ../../patches/moore/grounded-segment-anything-moore.patch
git -C thirdparty/GroundingDINO apply ../../patches/moore/groundingdino-moore.patch
git -C thirdparty/sam2 apply ../../patches/moore/sam2-moore.patch
```

The parent repository's vendored GroundingDINO compatibility changes are tracked
directly in the `moore` branch commit history; these files cover only submodule
working-tree changes.
