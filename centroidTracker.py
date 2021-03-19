from collections import OrderedDict
import cv2
import numpy as np
from imutils.video import FPS
import track_stats

# import the necessary packages
# from scipy.spatial import distance as dist
# import scipy.distance as dist


class CentroidTracker:
    def __init__(self, maxDisappeared=40, maxDistance=25):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.deregistered = []

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

        # store the maximum distance between centroids to associate
        # an object -- if the distance is larger than this maximum
        # distance we'll start to mark the object as "disappeared"
        self.maxDistance = maxDistance

    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # copy object content to deregistered dictionary before deletion
        self.deregistered.append(objectID)
        
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def get_distances (self, rows, columns):
        # result = np.sqrt(np.sum((objectCentroids - inputCentroids)**2,axis=1))
        # if len(D.shape) == 1:
        #     D = D.reshape(objectCentroids.shape[0],inputCentroids.shape[0])
        result = np.empty(shape=(rows.shape[0] * columns.shape[0]))
        index = 0
        for c in columns:
            for r in rows:
                result[index] = np.sqrt(np.sum((c - r)**2))
                index+=1
        return result.reshape((rows.shape[0], columns.shape[0]))

    def deduplicate (self, detections):
        uniques = []
        for arr in detections:
            if len(uniques) == 0:
                uniques.append(arr)
                continue
            if np.array_equal(arr, uniques[-1]):
                continue
            uniques.append(arr)
        return np.array(uniques)

    def update(self, detections):
        # reset currently deregistered objects
        self.deregistered = []

        # check to see if the list of input bounding box rectangles
        # is empty
        if len(detections) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects, self.deregistered

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(detections), 2), dtype="int")

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(detections):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            objectCentroids = np.array(objectCentroids)
            inputCentroids = np.array(inputCentroids)
            # ORIG
            # D = dist.cdist(objectCentroids, inputCentroids) 
        
            # Replaced based on https://stackoverflow.com/questions/60739883/substitute-scipy-spatial-distance
            # DOCS https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
            # D = np.sqrt(np.sum((objectCentroids - inputCentroids)**2,axis=1))
            # if len(D.shape) == 1:
            #     D = D.reshape(objectCentroids.shape[0],inputCentroids.shape[0])

            # MY
            D = self.get_distances(objectCentroids, inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()
            # rows = np.amin(D).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]
            # cols = [0]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                if row in usedRows or col in usedCols:
                    continue

                # if the distance between centroids is greater than
                # the maximum distance, do not associate the two
                # centroids to the same object
                if D[row, col] > self.maxDistance:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        # return the set of trackable objects
        return self.objects, self.deregistered


class PersonTracker:
    def __init__(self):
        self.ct = CentroidTracker(maxDisappeared=20, maxDistance=20)
        self.persons = {}
        self.fps = FPS()
        self.fps.start()
        self.last_person_direction = 'none'

    def parse(self, frame, detections):
        img_h = frame.shape[0]
        img_w = frame.shape[1]
        objects, deregistered_ids = self.ct.update([
            (int( x    * img_w), int( y    * img_h),
             int((x+w) * img_w), int((y+h) * img_h))
            for x, y, w, h in detections
        ])

        for object_id, centroid in objects.items():
            if np.array_equal(self.persons.get(object_id, [[]])[-1], centroid) == False:
                self.persons[object_id] = self.persons.get(object_id, []) + [centroid]

        deregistered_persons = []
        for object_id in deregistered_ids:
            deregistered_persons.append(self.persons.pop(object_id))

        return self.persons, deregistered_persons

    def get_directions(self):
        results = {
            "left": 0,
            "up": 0,
            "right": 0,
            "down": 0,
        }
        for person_id, centroids in self.persons.items():
            if len(centroids) < 10:
                continue
            x_list, y_list = [], []

            for centroid in centroids:
                x_list.append(centroid[0])
                y_list.append(centroid[1])

            x_max_min_diff = x_list.index(max(x_list)) - x_list.index(min(x_list))
            if x_max_min_diff > 0:
                results["right"] += 1
            else:
                results["left"] += 1

            y_max_min_diff = y_list.index(max(y_list)) - y_list.index(min(y_list))
            if y_max_min_diff > 0:
                results["down"] += 1
            else:
                results["up"] += 1

        return results


class PersonTrackerDebug(PersonTracker):
    def parse(self, frame, detections):
        persons, deregistered_persons = super().parse(frame, detections)

        self.fps.update()
        self.fps.stop()
        cv2.putText(frame, "FPS: {:03.0f}".format(self.fps.fps()), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), thickness=1)
        self.fps.start()
        if len(deregistered_persons) > 0:
            for deregistered_person in deregistered_persons:
                self.last_person_direction = track_stats.get_direction(deregistered_person, frame.shape)

        cv2.putText(frame, f"Directions {self.last_person_direction}", (10, 10), cv2.FONT_HERSHEY_PLAIN, 0.8,
            (0, 255, 0), 1)

        for object_id in persons:
            centroids = persons[object_id]
            cv2.putText(frame, f"ID {object_id} [{len(centroids)}]", (centroids[-1][0] - 10, centroids[-1][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.circle(frame, (centroids[-1][0], centroids[-1][1]), 4, (255, 255, 0), -1)

            if len(centroids) > 1:
                cv2.polylines(frame, np.int32([centroids]), isClosed=False, color=(0, 255, 0), thickness=1)


        return persons, deregistered_persons
