Following https://public.kitware.com/pipermail/paraview/2017-October/041077.html to visualize this export:
- load files in paraview as usual
---> multiple objects?
      - yes:  - select block1 and block2
              - add filter "Group Datasets" (Filters -> Common -> Group Datasets)
              - change coloring, if needed
              -> continue
      - no:   -> continue
- add filter "Warp By Vector" (Filters -> Common -> Warp By Vector) (to GroupDatasets-object or single object)
- select desired mode in WarpByVector -> Properties -> Vectors
- Time Manager (View -> Time Inspector to show window)
      - untik time sources
      - increase number of frames
      - Animations -> WrapByVector -> Scale Factor -> klick on "+"
      - edit this animation: Interpolation -> Sinusoid (Phase, Frequency, Offset as default)
      - set Value to desired amplitude (Value of Time 1 is not used)
- activate repeat and play animation
- show other modes by changing the vector in WarpByVector -> Properties -> Vectors

Adding object to GroupDataset:
- right click on the GroupDataset (or on one of the indicating objects)
- Change Input -> select other objects (use Crtl/Shift)