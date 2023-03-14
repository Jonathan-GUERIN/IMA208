import numpy as np
from kalman_filter import KalmanFilter


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0]) #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    Computes Intersection Over Union between two sets of bboxes in [x1,y1,x2,y2] format
    
    Arguments:
    bb_test: np.array(nb test boxes, 4)
    bb_gt: np.array(nb gt boxes, 4)
    
    Returns:
    IoU: np.array(nb test boxes, nb gt boxes), the IoU between all test and gt boxes
    """
    
    nb_test_boxes = bb_test.shape[0]
    nb_gt_boxes = bb_gt.shape[0]
    IoU = np.zeros((nb_test_boxes, nb_gt_boxes))
    for i in range(nb_test_boxes):
        for j in range(nb_gt_boxes):
            x1 = bb_test[i,0]
            x2 = bb_test[i,2]
            y1 = bb_test[i,1]
            y2 = bb_test[i,3]
            x3 = bb_gt[i,0]
            x4 = bb_gt[i,2]
            y3 = bb_gt[i,1]
            y4 = bb_gt[i,3]
            if x3 <= x2 and y3 <= y2 and x4 >= x1 and y4 >=y1:
                if x3 >= x1:
                    if y3 >= y1:
                        num = (x2-x3)*(y2-y3)
                    elif y3 < y1:
                        num = (x2-x3)*(y2-y1)
                elif x3 < x1:
                    if y3 >= y1:
                        num = (x2-x1)*(y2-y3)
                    elif y3 < y1:
                        num = (x2-x1)*(y2-y1)
                den = (x2-x1)*(y2-y1)+(x4-x3)*(y4-y3)
                IoU[i,j] = num/den
            else:
                IoU[i,j] = 0
    return IoU  


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the area and r is
    the aspect ratio
    """
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]

    x = int((x1+x2)/2)
    y = int((y1+y2)/2)
    s = int( (x2-x1)*(y2-y1))
    r = int((x2-x1)/(y2-y1))
    
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(bbox, score=None):
    """
    Takes a bounding box x in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    x = bbox[0]
    y = bbox[1]
    s = bbox[2]
    r = bbox[3]
    
    w = np.sqrt(r*s)
    l = np.sqrt(r/s)
    
    x1 = int(x-w//2)
    y1 = int(y-l//2)
    x2 = int(x+w//2)
    y2 = int(y+w//2)
    
    if(score==None):
        return np.array([x1, y1, x2, y2]).reshape((1,4))
    else:
        return np.array([x1, y1, x2, y2, score]).reshape((1,5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
        The state is in the form [[x,y,s,r,\dot{x},\dot{y},\dot{s}]]. 
        Only the first four dimensions are observed.
    """
    count = 0
    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
            bbox is in the [x1,y1,x2,y2] format
        """

        # define a constant velocity model
        x = bbox[0]
        y = bbox[1]
        s = bbox[2]
        r = bbox[3]
        dim_x = 7
        dim_z = 7
        self.kf = KalmanFilter(dim_x, dim_z)
            # Initialize a KalmanFilter with the correct dimension for the state and measurement
            
        # Initialize the state transition matrix and measurement matrix assuming a constant velocity model
        # with only location variables measured
        
        self.kf.F = np.eye(dim_x, dim_x) 
        self.kf.H = np.eye(dim_z, dim_x) 

        self.kf.P[4:,4:] *= 1000. # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.R[2:,2:] *= 10.

        self.kf.x = [[x,y,s,r,1,1,1]] 
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
            bbox is in the [x1,y1,x2,y2] format
        """
        
        x = bbox[0]
        y = bbox[1]
        s = bbox[2]
        r = bbox[3]
        
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        z = bbox
        R = self.kf.R
        H = self.kf.H
        
        self.kf.update(z, R, H) # write the call to the Kalman filter update

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
            
        F = self.kf.F
        Q = self.kf.Q
            
        self.predict(F,Q) # write the call to the Kalman filter predict
        
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold = 0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class MOT(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score], [x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections 
        (use np.empty((0, 5)) for frames without detections).
        Returns a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,5))
    