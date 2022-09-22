# trace generated using paraview version 5.10.0
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10
import pathlib

#### import the simple module from the paraview
from paraview.simple import *

msh_stppvd = PVDReader(registrationName='msh_stp.pvd', FileName='/Users/jacobmerson/Documents/FCL-Results/thin-model/diluted_delaunay_bone_uniax_x_7.5_6_2196545/msh_stp.pvd')

filePath = pathlib.Path(__file__).parent.resolve()
pythonPath = str(filePath)

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

UpdatePipeline(time=0.0, proxy=msh_stppvd)
#SetActiveSource(msh_stppvd)
extractTimeSteps1 = ExtractTimeSteps(registrationName='ExtractTimeSteps1', Input=msh_stppvd)
extractTimeSteps1.TimeStepIndices = [0, 1, 2]
UpdatePipeline(time=0, proxy=extractTimeSteps1)
# create a new 'Slice'
slice1 = Slice(registrationName='Slice1', Input=extractTimeSteps1)

# Properties modified on slice1.SliceType
slice1.SliceType.Origin = [-0.07944972233201164, 0.0005, -0.013754037567371903]
slice1.SliceType.Normal = [0.0, 1.0, 0.0]

UpdatePipeline(time=0.0, proxy=slice1)

# create a new 'Resample To Image'
resampleToImage1 = ResampleToImage(registrationName='ResampleToImage1', Input=slice1)
# Properties modified on resampleToImage1
resampleToImage1.SamplingDimensions = [50, 1, 50]

UpdatePipeline(time=0.0, proxy=resampleToImage1)

# create a new 'Programmable Filter'
alignmentAndOrientation = ProgrammableFilter(registrationName='AlignementAndOrientation', Input=resampleToImage1)
alignmentAndOrientation.Script = """
import vectorcorrelation
# force reload of vectorcorrelation module
import importlib; importlib.reload(vectorcorrelation.paraview)
vectorcorrelation.paraview.alignment_and_orientation(inputs, output)
"""
alignmentAndOrientation.PythonPath = pythonPath
UpdatePipeline(time=0.0, proxy=alignmentAndOrientation)


# create a new 'Group Time Steps'
groupTimeSteps1 = GroupTimeSteps(registrationName='GroupTimeSteps1', Input=alignmentAndOrientation)
UpdatePipeline(time=0.0, proxy=groupTimeSteps1)
vectorCorrelation = ProgrammableFilter(registrationName='VectorCorrelation',Input=groupTimeSteps1)
vectorCorrelation.Script = """
import vectorcorrelation
# force reload of vectorcorrelation module
import importlib; importlib.reload(vectorcorrelation.paraview)
vectorcorrelation.paraview.correlation(inputs, output)
"""
vectorCorrelation.PythonPath = pythonPath
vectorCorrelation.CopyArrays = True

UpdatePipeline(time=0.0, proxy=vectorCorrelation)

extractBlock1 = ExtractBlock(registrationName='ExtractBlock1', Input=vectorCorrelation)
extractBlock1.Selectors = ['/Root/timestep2']
UpdatePipeline(time=0, proxy=extractBlock1)
