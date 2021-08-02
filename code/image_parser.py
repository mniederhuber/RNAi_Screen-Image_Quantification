import numpy as np
import pandas as pd
from aicsimageio import AICSImage

from skimage import filters, io, morphology, feature, img_as_ubyte
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
    'size' : [],
    'stack' : [],
		'gfp':[],
    'brDisc':[],
    'brDisc_gfp' : [],
    'brDisc_gfpNEG' : [],
    'KDtoWT' : []
}

n = 0
for sample_id in sample_sheet_LL['id']:
    percent_complete = round(100*(n/len(sample_sheet_LL.id)), 2)
    print(f"{percent_complete}% complete")

    line = sample_sheet_LL.loc[sample_sheet_LL['id'] == sample_id, 'line'].iloc[0]
    date = sample_sheet_LL.loc[sample_sheet_LL['id'] == sample_id, 'date'].iloc[0]
    fp = sample_sheet_LL.loc[sample_sheet_LL['id'] == sample_id, 'fp'].iloc[0]
    symbol = sample_sheet_LL.loc[sample_sheet_LL['id'] == sample_id, 'Symbol'].iloc[0]
    
    im = AICSImage(fp)
    
    #print(im)
    #print(im.scenes)
    for scene in im.scenes:
      #  print(sample_id, scene)
        im.set_scene(scene)

        current_scene = im.current_scene
        shape = im.shape
        array = im.data
        sample = sample_id
        stack = im.shape[2]        
        size = f"{im.shape[3]}x{im.shape[4]}"

        if im.shape[1] == 3:
            dapi = np.amax(im.data[:,0,:,:,:], axis = 1)
            gfp = np.amax(im.data[:,1,:,:,:], axis = 1)
            brDisc = np.amax(im.data[:,2,:,:,:], axis = 1)
            
           # print(dapi.shape)
            dapi_gauss = filters.gaussian(dapi, sigma = 5)
            dapi_mask = dapi_gauss > 0.1
            #dapi_mask = dapi_gauss >= filters.threshold_otsu(dapi_gauss)

            gfp_gauss = filters.gaussian(gfp, sigma = 5)
            gfp_cut = np.where(dapi_mask == False, 0, gfp_gauss)
            gfp_mask = gfp_cut >= 0.1 
            #gfp_mask = gfp_gauss >= filters.threshold_otsu(gfp_gauss)             

            gfpNEG_mask = np.array(np.subtract(gfp_mask, dapi_mask, dtype = float), dtype = bool)

            brDisc_gfp = np.zeros_like(brDisc)
            brDisc_gfp[gfp_mask] = brDisc[gfp_mask]
            brDisc_gfp_signal = np.mean(brDisc_gfp)

            brDisc_gfpNEG = np.zeros_like(brDisc)
            brDisc_gfpNEG[gfpNEG_mask] = brDisc[gfpNEG_mask]
            brDisc_gfpNEG_signal = np.mean(brDisc_gfpNEG)

            KDtoWT = brDisc_gfp_signal / brDisc_gfpNEG_signal
            
            io.imsave(f"output/{sample}_{scene}_GFP_MAX.tiff", gfp)
            io.imsave(f"output/{sample}_{scene}_GFP-MASK_MAX.tiff", img_as_ubyte(gfp_mask))
            io.imsave(f"output/{sample}_{scene}_GFPNEG-MASK_MAX.tiff", img_as_ubyte(gfpNEG_mask))
            io.imsave(f"output/{sample}_{scene}_DAPI_MAX.tiff", dapi)
            io.imsave(f"output/{sample}_{scene}_DAPI-MASK_MAX.tiff", img_as_ubyte(dapi_mask))
            io.imsave(f"output/{sample}_{scene}_brDisc_MAX.tiff", brDisc)
            
            outdata['symbol'].append(symbol)
            outdata['line'].append(line)
            outdata['date'].append(date)
            outdata['scene'].append(scene)
            outdata['shape'].append(shape)
            outdata['size'].append(size)
            outdata['stack'].append(stack)
            outdata['gfp'].append(np.mean(gfp))
            outdata['brDisc'].append(np.mean(brDisc))
            outdata['brDisc_gfp'].append(brDisc_gfp_signal)
            outdata['brDisc_gfpNEG'].append(brDisc_gfpNEG_signal)
            outdata['KDtoWT'].append(KDtoWT)

    n = n + 1

df = pd.DataFrame(outdata)
pd.DataFrame.to_csv(df, 'output/outdata.csv')





