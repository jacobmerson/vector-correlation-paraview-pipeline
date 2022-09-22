
def alignment_and_orientation(inputs, output):
    import numpy as np
    input0 = inputs[0]
    #print(dir(input0))
    omega = input0.PointData["ornt_tens_2D_1"]

    mask = np.isclose(omega,0)[:,0,0]

    # max alignment in plane (from 2D orientation tensor)
    alignment = np.max(np.linalg.eigvals(omega),axis=-1)

    theta_xx = np.arccos(np.sqrt(omega[:,0,0]))
    theta_xx[mask] = 1
    alignment[mask] = 1


    #print(ornt_tens.shape)
    output.PointData.append(theta_xx, "angle")
    output.PointData.append(alignment, "alignment")

def correlation(inputs, output):
    import vectorcorrelation.analysis as veccorr
    import numpy as np
    input0 = inputs[0]
    #alignments = input0.PointData['alignment']
    analysis = veccorr.Analysis()
    for block in input0:
        alignment = block.PointData["alignment"]
        angle = block.PointData["angle"]
        alignment = alignment.reshape(int(np.sqrt(alignment.shape[0])),-1)
        angle = angle.reshape(int(np.sqrt(angle.shape[0])),-1)
        assert(alignment.shape == angle.shape)
        analysis.add_frame(angle,alignment)
    analysis.run(threshold=0.1, blocksize=3)
    num_correlations = len(analysis.correlation_images)
    num_diffs = len(analysis.diff_images)

    for i,block in enumerate(output):
        if i>=1:
            block.PointData.append(analysis.correlation_images[i-1].ravel(),"correlation")
        if i>=2:
            block.PointData.append(analysis.diff_images[i-2].ravel(),"diff")
