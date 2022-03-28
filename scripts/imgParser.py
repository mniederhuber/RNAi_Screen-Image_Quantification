import os
import numpy as np
import pandas as pd
from aicsimageio import AICSImage
import pickle
from pathlib import Path
import glob

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
	for root, dirs, files in os.walk(imgDir):
		for f in files:
			if '.lif' in f and '.lifext' not in f and 'E93' not in f:
				if '_' in root:
					fp = os.path.abspath(os.path.join(root,f))
					date = root.split('/')[1].split('_')[0]
					line = f.split('_')[0]
					symbol = lineLUT[line]
					sampleID = '_'.join([symbol,line,date])
					image = AICSImage(fp)
					yield {
						'line' : line,
						'symbol' : symbol,
						'date' : date,
						'sampleID' : sampleID,
						'fp' : fp,
						'image' : image,
					}
				elif '_' in f:
					fp = os.path.abspath(os.path.join(root,f))
					date = root.split('/')[1]
					line = f.split('_')[0]
					row = [line,date,fp]
					symbol = lineLUT[line]
					sampleID = '_'.join([symbol,line,date])
					image = AICSImage(fp)
					yield {
						'line' : line,
						'symbol' : symbol,
						'date' : date,
						'sampleID' : sampleID,
						'fp' : fp,
						'image' : image,
					}
				else:
					fp = os.path.abspath(os.path.join(root,f))
					date = root.split('/')[1]
					line = f.split('.')[0]
					row = [line,date,fp]
					symbol = lineLUT[line]
					sampleID = '_'.join([symbol,line,date])
					image = AICSImage(fp)
					yield {
						'line' : line,
						'symbol' : symbol,
						'date' : date,
						'sampleID' : sampleID,
						'fp' : fp,
						'image' : image,
					}

def getScene(metas):
		for sample in metas:
				for scene in sample['image'].scenes:
						sample['scene'] = scene
						yield sample

def quantify(samples):
		for sample in samples:
				image = sample['image']
				scene = sample['scene']
				image.set_scene(scene)	
				shape = image.shape	
				array = image.data
				sampleID = sample['sampleID']
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
								'symbol' : sample['symbol'],
								'line' : sample['line'],
								'date' : sample['date'],
								'sampleID' : sample['sampleID'],
								'scene' : scene,
								'shape' : '.'.join(map(str,shape)),
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

def writeOut(results):
		for result in results:
				io.imsave(f"output/maxProj/{result['sampleID']}_{result['scene']}_GFP_MAX.tiff", result['GFP_MAX'])
				io.imsave(f"output/mask/{result['sampleID']}_{result['scene']}_GFP-MASK_MAX.tiff", result['GFP-mask'])
				io.imsave(f"output/mask/{result['sampleID']}_{result['scene']}_GFPNEG-MASK_MAX.tiff", result['GFPneg-mask'])
				io.imsave(f"output/maxProj/{result['sampleID']}_{result['scene']}_DAPI_MAX.tiff", result['DAPI_MAX'])
				io.imsave(f"output/mask/{result['sampleID']}_{result['scene']}_DAPI-MASK_MAX.tiff", result['DAPI-mask'])
				io.imsave(f"output/maxProj/{result['sampleID']}_{result['scene']}_brDisc_MAX.tiff", result['brDisc_MAX'])
				
#				with open(f"output/pickles/{out['sampleID']}_{out['scene']}.pkl", 'wb') as f:
#						pickle.dump({k:out[k] for k in ('symbol', 'line', 'date', 'sampleID', 'scene', 'shape', 'size', 'stack', 'gfp', 'brDisc', 'brDisc_gfp', 'brDisc_gfpNEG', 'KDtoWT')}, f)
				yield {k:result[k] for k in ('symbol', 'line', 'date', 'sampleID', 'scene', 'shape', 'size', 'stack', 'gfp', 'brDisc', 'brDisc_gfp', 'brDisc_gfpNEG', 'KDtoWT')}
				

def concatDF(outs):
		dflist = []
		for out in outs:
				dflist.append(out)
		pd.DataFrame(dflist).to_csv('output/resultsDF.csv', index = False)

lineLUT = makeLineDic(input_sheet)
#chain the generators
meta = getMeta(image_directory, lineLUT)
scenes = getScene(meta)
results = quantify(scenes)
outputs = writeOut(results)
concatDF(outputs)
#list(concatDF)	

