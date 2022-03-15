import os
import numpy as np
import pandas as pd
from aicsimageio import AICSImage

from skimage import filters, io, morphology, feature, img_as_ubyte
from skimage.measure import label, regionprops
from tifffile import imsave, imread

image_directory = 'imgsTEMP'
input_sheet = 'sheets/results.csv'

def makeLineDic(sheet):
	lut = {}
	s = pd.read_csv(sheet)
	for row in s.itertuples(index = False):
		lut[row.Line] = row.Symbol	
	return lut

def getMeta(imgDir, lineLUT):
	print('getMeta')
	for root, dirs, files in os.walk(imgDir):
		for f in files:
			if '.lif' in f and '.lifext' not in f and 'E93' not in f:
				if '_' in root:
					fp = os.path.abspath(os.path.join(root,f))
					date = root.split('/')[1].split('_')[0]
					line = f.split('_')[0]
					symbol = lineLUT[line]
					sampleID = '_'.join([symbol,line,date])
					yield {
						'line' : line,
						'symbol' : symbol,
						'date' : date,
						'sampleID' : sampleID,
						'fp' : fp
					}
				elif '_' in f:
					fp = os.path.abspath(os.path.join(root,f))
					date = root.split('/')[1]
					line = f.split('_')[0]
					row = [line,date,fp]
					symbol = lineLUT[line]
					sampleID = '_'.join([symbol,line,date])
					yield {
						'line' : line,
						'symbol' : symbol,
						'date' : date,
						'sampleID' : sampleID,
						'fp' : fp
					}
				else:
					fp = os.path.abspath(os.path.join(root,f))
					date = root.split('/')[1]
					line = f.split('.')[0]
					row = [line,date,fp]
					symbol = lineLUT[line]
					sampleID = '_'.join([symbol,line,date])
					yield {
						'line' : line,
						'symbol' : symbol,
						'date' : date,
						'sampleID' : sampleID,
						'fp' : fp
					}

def loadLif(meta):
		print('loadLif')
		for lif in meta:
				yield AICSImage(lif['fp'])

def getScenes(lifs):
		print('getScenes')
		for lif in lifs:
				yield lif.scenes 

def setScene(scenes):
		for scene in scenes:
				yield scene

def quantify(scenes, image, meta):
		for scene in scenes:
				image.set_scene(scene)
				shape = image.shape	
				array = image.data
				sample = meta['sample_ID']
				stack = image.shape[2]        
				size = f"{image.shape[3]}x{image.shape[4]}"
		
		#check that scene has all 3 channels (DAPI, GFP, tdTomato)
				if image.shape[1] == 3:
		
		#make Zmax projections of each channel
						dapi = np.amax(image.data[:,0,:,:,:], axis = 1)
						gfp = np.amax(image.data[:,1,:,:,:], axis = 1)
						brDisc = np.amax(image.data[:,2,:,:,:], axis = 1)
						
						dapi_gauss = filters.gaussian(dapi, sigma = 5)
		#make a boolean mask of DAPI -- 0.1 was empirically chosen by testing in FIJI
						dapi_mask = dapi_gauss > 0.1
		
						gfp_gauss = filters.gaussian(gfp, sigma = 5)
		#cut out parts of GFP image where there is no DAPI
						gfp_cut = np.where(dapi_mask == False, 0, gfp_gauss)
		
		#make GFP mask, see above re: 0.1 cutoff
						gfp_mask = gfp_cut >= 0.1 
						#gfp_mask = gfp_gauss >= filters.threshold_otsu(gfp_gauss)             
		
		#make an inverse mask to GFP mask -- this serves as a mask for the WT part of the wing 
						gfpNEG_mask = np.array(np.subtract(gfp_mask, dapi_mask, dtype = float), dtype = bool)
		
		#get mean values of brDisc signal in the GFP masked region
						brDisc_gfp = np.zeros_like(brDisc)
						brDisc_gfp[gfp_mask] = brDisc[gfp_mask]
						brDisc_gfp_signal = np.mean(brDisc_gfp)
		
		#get mean values of brDisc signal in the GFP-negative masked region
						brDisc_gfpNEG = np.zeros_like(brDisc)
						brDisc_gfpNEG[gfpNEG_mask] = brDisc[gfpNEG_mask]
						brDisc_gfpNEG_signal = np.mean(brDisc_gfpNEG)
		
		#calculate the knockdown to WT ratio
						KDtoWT = brDisc_gfp_signal / brDisc_gfpNEG_signal
	
						yield {
								'symbol' : meta['symbol'],
								'line' : meta['line'],
								'date' : meta['date'],
								'sample' : meta['sampleID'],
								'scene' : scene,
								'shape' : shape,
								'size' : size,
								'stack' : stack,
								'gfp' : np.mean(gfp),
								'brDisc' : np.mean(brDisc),
								'brDisc_gfp' : brDisc_gfp_signal,
								'brDisc_gfpNEG' : brDisc_gfpNEG_signal,
								'KDtoWT' : KDtoWT,
								'brDisc_MAX' : brDisc,
								'GFP_MAX' : gfp,
								'DAPI_MAX' : dapi,
								'GFP-mask' : img_as_ubyte(gfp_mask),
								'GFPneg-mask' : img_as_ubyte(gfpNEG_mask),
								'DAPI-mask' : img_as_ubyte(dapi_mask)
						}

def writeOut(outputs):
		for out in outputs:
#				io.imsave(f"output/maxProj/{out['sample']}_{out['scene']}_GFP_MAX.tiff", out['GFP_MAX'])
#        io.imsave(f"output/mask/{out['sample']}_{out['scene']}_GFP-MASK_MAX.tiff", out['GFP-mask'])
#        io.imsave(f"output/mask/{out['sample']}_{out['scene']}_GFPNEG-MASK_MAX.tiff", out['GFPneg-mask')
#        io.imsave(f"output/maxProj/{out['sample']}_{out['scene']}_DAPI_MAX.tiff", out['DAPI_MAX'])
#        io.imsave(f"output/mask/{out['sample']}_{out['scene']}_DAPI-MASK_MAX.tiff", out['DAPI-mask'])
#        io.imsave(f"output/maxProj/{out['sample']}_{out['scene']}_brDisc_MAX.tiff", out['brDisc_MAX'])
				
				with open(f"output/pickles/{out['sample']}.pkl", 'wb') as f:
						pickle.dump({k:out[k] for k in ('symbol', 'line', 'date', 'sample', 'scene', 'shape', 'size', 'stack', 'gfp', 'brDisc', 'brDisc_gfp', 'brDisc_gfpNEG', 'KDtoWT')}, f)
				


lineLUT = makeLineDic(input_sheet)
meta = getMeta(image_directory, lineLUT)
lifs = loadLif(meta)
scenes = getScenes(lifs)
lifscene = setScene(scenes)
outputs = quantify(scenes, lifs, meta)
for x in lifs:
	print(x)
for x in scenes:
	print(x)
for x in lifscene:
	prin(x)
#list(outputs)	

#    #output max projections and masks to check accuracy
#            io.imsave(f"output/maxProj/{sample}_{scene}_GFP_MAX.tiff", gfp)
#            io.imsave(f"output/mask/{sample}_{scene}_GFP-MASK_MAX.tiff", img_as_ubyte(gfp_mask))
#            io.imsave(f"output/mask/{sample}_{scene}_GFPNEG-MASK_MAX.tiff", img_as_ubyte(gfpNEG_mask))
#            io.imsave(f"output/maxProj/{sample}_{scene}_DAPI_MAX.tiff", dapi)
#            io.imsave(f"output/mask/{sample}_{scene}_DAPI-MASK_MAX.tiff", img_as_ubyte(dapi_mask))
#            io.imsave(f"output/maxProj/{sample}_{scene}_brDisc_MAX.tiff", brDisc)
#            
#    #output values to the outdata dictionary
#            outdata['symbol'].append(symbol)
#            outdata['line'].append(line)
#            outdata['date'].append(date)
#            outdata['scene'].append(scene)
#            outdata['shape'].append(shape)
#            outdata['size'].append(size)
#            outdata['stack'].append(stack)
#            outdata['gfp'].append(np.mean(gfp))
#            outdata['brDisc'].append(np.mean(brDisc))
#            outdata['brDisc_gfp'].append(brDisc_gfp_signal)
#            outdata['brDisc_gfpNEG'].append(brDisc_gfpNEG_signal)
#            outdata['KDtoWT'].append(KDtoWT)
#
#outdata dictionary
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


#quantify(getPaths(image_directory,lineLUT))

#for sample_id in imgDF['sampleID']:
#    #percent_complete = round(100*(n/len(imgDF.sampleID)), 2)
#    #print(f"{percent_complete}% complete")
#
#    #get image id variables
#    line = imgDF.loc[imgDF['sampleID'] == sample_id, 'Line'].iloc[0]
#    date = imgDF.loc[imgDF['sampleID'] == sample_id, 'Date'].iloc[0]
#    fp = imgDF.loc[imgDF['sampleID'] == sample_id, 'fPath'].iloc[0]
#    symbol = imgDF.loc[imgDF['sampleID'] == sample_id, 'Symbol'].iloc[0]
#	
#    #load image 
#    im = AICSImage(fp)
#   
#    for scene in im.scenes:
#
#    #AICSImage will pull the data of the current scene only -- need to specify scene  
#        im.set_scene(scene)
#
#    #get scene values
#        shape = im.shape
#        array = im.data
#        sample = sample_id
#        stack = im.shape[2]        
#        size = f"{im.shape[3]}x{im.shape[4]}"
#
#    #check that scene has all 3 channels (DAPI, GFP, tdTomato)
#        if im.shape[1] == 3:
#
#    #make Zmax projections of each channel
#            dapi = np.amax(im.data[:,0,:,:,:], axis = 1)
#            gfp = np.amax(im.data[:,1,:,:,:], axis = 1)
#            brDisc = np.amax(im.data[:,2,:,:,:], axis = 1)
#            
#            dapi_gauss = filters.gaussian(dapi, sigma = 5)
#    #make a boolean mask of DAPI -- 0.1 was empirically chosen by testing in FIJI
#            dapi_mask = dapi_gauss > 0.1
#
#            gfp_gauss = filters.gaussian(gfp, sigma = 5)
#    #cut out parts of GFP image where there is no DAPI
#            gfp_cut = np.where(dapi_mask == False, 0, gfp_gauss)
#
#    #make GFP mask, see above re: 0.1 cutoff
#            gfp_mask = gfp_cut >= 0.1 
#            #gfp_mask = gfp_gauss >= filters.threshold_otsu(gfp_gauss)             
#
#    #make an inverse mask to GFP mask -- this serves as a mask for the WT part of the wing 
#            gfpNEG_mask = np.array(np.subtract(gfp_mask, dapi_mask, dtype = float), dtype = bool)
#
#    #get mean values of brDisc signal in the GFP masked region
#            brDisc_gfp = np.zeros_like(brDisc)
#            brDisc_gfp[gfp_mask] = brDisc[gfp_mask]
#            brDisc_gfp_signal = np.mean(brDisc_gfp)
#
#    #get mean values of brDisc signal in the GFP-negative masked region
#            brDisc_gfpNEG = np.zeros_like(brDisc)
#            brDisc_gfpNEG[gfpNEG_mask] = brDisc[gfpNEG_mask]
#            brDisc_gfpNEG_signal = np.mean(brDisc_gfpNEG)
#
#    #calculate the knockdown to WT ratio
#            KDtoWT = brDisc_gfp_signal / brDisc_gfpNEG_signal
#            
#    #output max projections and masks to check accuracy
#            io.imsave(f"output/maxProj/{sample}_{scene}_GFP_MAX.tiff", gfp)
#            io.imsave(f"output/mask/{sample}_{scene}_GFP-MASK_MAX.tiff", img_as_ubyte(gfp_mask))
#            io.imsave(f"output/mask/{sample}_{scene}_GFPNEG-MASK_MAX.tiff", img_as_ubyte(gfpNEG_mask))
#            io.imsave(f"output/maxProj/{sample}_{scene}_DAPI_MAX.tiff", dapi)
#            io.imsave(f"output/mask/{sample}_{scene}_DAPI-MASK_MAX.tiff", img_as_ubyte(dapi_mask))
#            io.imsave(f"output/maxProj/{sample}_{scene}_brDisc_MAX.tiff", brDisc)
#            
#    #output values to the outdata dictionary
#            outdata['symbol'].append(symbol)
#            outdata['line'].append(line)
#            outdata['date'].append(date)
#            outdata['scene'].append(scene)
#            outdata['shape'].append(shape)
#            outdata['size'].append(size)
#            outdata['stack'].append(stack)
#            outdata['gfp'].append(np.mean(gfp))
#            outdata['brDisc'].append(np.mean(brDisc))
#            outdata['brDisc_gfp'].append(brDisc_gfp_signal)
#            outdata['brDisc_gfpNEG'].append(brDisc_gfpNEG_signal)
#            outdata['KDtoWT'].append(KDtoWT)
#
#    n = n + 1
#
##build the outdata dataframe and save
#df = pd.DataFrame(outdata)
#pd.DataFrame.to_csv(df, 'output/outdata.csv')
#
#
#
#
#
