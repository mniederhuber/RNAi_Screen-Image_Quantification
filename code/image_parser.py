import numpy as np
import pandas as pd
from aicsimageio import AICSImage

from skimage import filters, io, morphology, feature
from skimage.measure import label, regionprops
from tifffile import imsave, imread




sample_sheet_LL = pd.read_csv('sheets/sample_sheet.tsv', sep = '\t')
sample_sheet = pd.read_csv('sheets/results_fp.csv')

sample_sheet['date'] = sample_sheet['date'].fillna(0).astype(int)
sample_sheet['date'] = sample_sheet['date'].astype(int)

symbol_dict = {}

sample_id = []
for x, y in sample_sheet.iterrows():
    symbol = y['Symbol']
    line = y['line']
    date = str(y['date'])
    
    sample_id.append('_'.join([symbol,line,date]))
    symbol_dict[line] = symbol
    
sample_sheet['id'] = sample_id  
sample_sheet_LL['Symbol'] = sample_sheet_LL['line'].map(symbol_dict)

sample_id = []
for x, y in sample_sheet_LL.iterrows():
    #print(x)
    symbol = y['Symbol']
    line = y['line']
    date = str(y['date'])
    
    sample_id.append('_'.join([symbol,line,date]))
    
sample_sheet_LL['id'] = sample_id  


outdata = {
    'symbol' : [],
    'line' : [],
    'date' : [],
    'scene' : [],
    'shape' : [],
    'stack' : [],
    'brDisc_gfp' : [],
    'brDisc_gfpNEG' : [],
    'KDtoWT' : []
}

for sample_id in sample_sheet_LL['id']:
    line = sample_sheet_LL.loc[sample_sheet_LL['id'] == sample_id, 'line'].iloc[0]
    date = sample_sheet_LL.loc[sample_sheet_LL['id'] == sample_id, 'date'].iloc[0]
    fp = sample_sheet_LL.loc[sample_sheet_LL['id'] == sample_id, 'fp'].iloc[0]
    symbol = sample_sheet_LL.loc[sample_sheet_LL['id'] == sample_id, 'Symbol'].iloc[0]
    
    im = AICSImage(fp)
    
    #print(im)
    #print(im.scenes)
    for scene in im.scenes:
        print(sample_id, scene)
        im.set_scene(scene)

        current_scene = im.current_scene
        shape = im.shape
        array = im.data
        sample = sample_id
        stack = im.shape[2]        
        if im.shape[1] == 3:
            dapi = np.max(im.data[:,0,:,:,:], axis = 0)
            gfp = np.max(im.data[:,1,:,:,:], axis = 0)
            brDisc = np.max(im.data[:,2,:,:,:], axis = 0)

            dapi_gauss = filters.gaussian(dapi, sigma = 4)
            dapi_mask = dapi_gauss >= filters.threshold_otsu(dapi_gauss)

            gfp_gauss = filters.gaussian(gfp, sigma = 4)
            gfp_mask = gfp_gauss >= filters.threshold_otsu(gfp_gauss)             

            gfpNEG_mask = np.array(np.subtract(dapi_mask, gfp_mask, dtype = np.float), dtype = bool)

            brDisc_gfp = np.zeros_like(brDisc)
            brDisc_gfp[gfp_mask] = brDisc[gfp_mask]
            brDisc_gfp_signal = np.mean(brDisc_gfp)

            brDisc_gfpNEG = np.zeros_like(brDisc)
            brDisc_gfpNEG[gfpNEG_mask] = brDisc[gfpNEG_mask]
            brDisc_gfpNEG_signal = np.mean(brDisc_gfpNEG)

            KDtoWT = brDisc_gfp_signal / brDisc_gfpNEG_signal

            io.imsave(f"output/{sample}_{scene}_GFP-MASK_MAX.tiff", gfp_mask)
            io.imsave(f"output/{sample}_{scene}_GFPNEG-MASK_MAX.tiff", gfpNEG_mask)
            io.imsave(f"output/{sample}_{scene}_DAPI-MASK_MAX.tiff", dapi_mask)
            
            outdata['symbol'].append(symbol)
            outdata['line'].append(line)
            outdata['date'].append(date)
            outdata['scene'].append(scene)
            outdata['shape'].append(shape)
            outdata['stack'].append(stack)
            outdata['brDisc_gfp'].append(brDisc_gfp_signal)
            outdata['brDisc_gfpNEG'].append(brDisc_gfpNEG_signal)
            outdata['KDtoWT'].append(KDtoWT)

df = pd.DataFrame(outdata)
pd.DataFrame.to_csv(df, 'output/outdata.csv')










