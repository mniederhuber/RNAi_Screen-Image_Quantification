Workflow


~~- generate `sample_sheet.tsv` in the `sheets/` dir, using `sheet_gen.py` to build sheet with ids and filepaths to all .lif files in the working directory~~
- workflow requires an input sheet `.csv` that contains columns for bloomington/VDRC Line number and corresponding Gene Symbol
```
Line,Symbol,...
31266,osa,...
34827,dom,...
```
- Edit `imageParser.py` to specify the `input sheet` path and `image directory` path
- Run `imgParser.py` 
