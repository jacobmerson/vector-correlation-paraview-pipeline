import numpy as np

def alignment_and_orientation(inputs, output):
    input0 = inputs[0]
    #print(dir(input0))
    omega = input0.PointData["ornt_tens_2D_1"]

    mask = np.isclose(omega,0)[:,0,0]

    # max alignment in plane (from 2D orientation tensor)
    #alignment = np.max(np.linalg.eigvals(omega),axis=-1)
    eigval, eigvec = np.linalg.eig(omega)
    idx_eigmax = np.argmax(eigval, axis=-1)

    alignment = np.empty_like(idx_eigmax, dtype=np.float64)
    theta = np.empty_like(idx_eigmax, dtype=np.float64)
    #max_eigvec = np.empty((len(idx_eigmax),3))
    for i in range(len(idx_eigmax)):
        alignment[i] = eigval[i,idx_eigmax[i]]
        max_eigvec = eigvec[i,:,idx_eigmax[i]]
        #theta[i] = np.arctan2(max_eigvec[0], max_eigvec[2])
        theta[i] = np.arctan2(max_eigvec[2], max_eigvec[0])

    output.PointData.append(theta, "angle")
    output.PointData.append(alignment, "alignment")

def correlation(inputs, output):
    import vectorcorrelation.analysis as veccorr
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

    alignment = input0.PointData["alignment"].Arrays
    angle = input0.PointData["angle"].Arrays
    for i,block in enumerate(output):
        if i>=1:
            block.PointData.append((alignment[i] - alignment[i-1]), "alignment difference")
            block.PointData.append((angle[i] - angle[i-1]), "angle difference")
            block.PointData.append(analysis.correlation_images[i-1].ravel(),"correlation")
        if i>=2:
            # (f1-f0)-(f2-f1)
            block.PointData.append((-2*alignment[i-1] + alignment[i-2] + alignment[i]), "alignment diff diff")
            block.PointData.append((-2*angle[i-1] + angle[i-2] + angle[i]), "angle diff diff")


            block.PointData.append(analysis.diff_images[i-2].ravel(),"diff")


##https://stackoverflow.com/questions/28342968/how-to-plot-a-2d-gaussian-with-different-sigma
def multivariate_normal(x, mean, covariance):
    assert(x.shape == mean.shape)
    assert(x.shape[0] == covariance.shape[0])
    assert(x.shape[0] == covariance.shape[1])
    n = mean.shape[0]
    inv_covariance = np.linalg.inv(covariance)
    det_covariance = np.linalg.det(covariance)
    denom = np.sqrt((2*np.pi)**n *det_covariance)
    num = np.exp(-0.5*(x-mean).T @ inv_covariance @ (x-mean))
    return num/denom

def GetUpdateTimestep(algorithm):
    """Returns the requested time value, or None if not present"""
    executive = algorithm.GetExecutive()
    outInfo = executive.GetOutputInformation(0)
    return outInfo.Get(executive.UPDATE_TIME_STEP()) \
              if outInfo.Has(executive.UPDATE_TIME_STEP()) else None

def surrogate_orientation(algorithm, inputs, output):
    input0 = inputs[0]
    orientation_tensor = np.zeros_like(input0.CellData["ornt_tens_2D_1"])
    #angle = np.random.vonmises(0, 1.0, orientation_tensor.shape[0])
    angle = np.zeros(len(orientation_tensor))
    
    


    bounds = input0.GetBounds()
    center =np.array([(bounds[1]+bounds[0])/2.0, (bounds[4]+bounds[5])/2.0])


    time = GetUpdateTimestep(algorithm)
    #sigma = np.eye(2,dtype=np.float64)/5000000.
    sigma = np.eye(2,dtype=np.float64)/1000000.
    #if time >1.0:
        #sigma /= 5.0
    mu = center
    radius = 2E-3
    height = 0.9*0.6
    theta = np.arctan2(height,radius)

    if time > 1.1:
        height = 0.6
        # adjust the radius to keep the same slope
        radius = height/np.tan(theta)

    numTets = input0.GetNumberOfCells()
    assert(orientation_tensor.shape[0] == numTets)

    centroid = np.empty(3, dtype=np.float64)
    alignment = np.zeros(numTets, dtype=np.float64)

    # gaussian case
    for i in range(numTets):
        cell = input0.GetCell(i)
        cell.GetCentroid(centroid)
        x = np.array((centroid[0], centroid[2]))
        alignment[i] = multivariate_normal(x, mu, sigma)
    alignment *= height/np.max(alignment)

    # cone case
    #for i in range(numTets):
    #    cell = input0.GetCell(i)
    #    cell.GetCentroid(centroid)
    #    r = np.array((centroid[0], centroid[2]))
    #    r -= center
    #    r = np.linalg.norm(r)
    #    if r < radius:
    #        q = radius-r
    #        alignment[i] = q*np.tan(theta)
    #    else:
    #        alignment[i] = 0

    # max eigenvalue of orientation tensor must be between 0 and 1
    assert(np.max(alignment) <= 1.0)
    assert(np.min(alignment) >= 0)
    
    #for i in range(numTets):
    orientation_tensor[:,0,0] = alignment*np.cos(angle)**2
    orientation_tensor[:,0,2] = alignment*np.cos(angle)*np.sin(angle)
    orientation_tensor[:,2,0] = alignment*np.cos(angle)*np.sin(angle)
    orientation_tensor[:,2,2] = alignment*np.sin(angle)**2


    #mask = np.bitwise_not(mask)
    #orientation_tensor[mask,0,0] = field[mask]
    #orientation_tensor[mask,1,1] = (1-field[mask])/2.0
    #orientation_tensor[mask,2,2] = (1-field[mask])/2.0
    #output.CellData.append(field,"gaussian")
    #output.CellData.append(orientation_tensor, "orientation tensor")
    #orientation_tensor = input0.CellData["ornt_tens_2D_1"]
    output.CellData.append(orientation_tensor, "ornt_tens_2D_1")

