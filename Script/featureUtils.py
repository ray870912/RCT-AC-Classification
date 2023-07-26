import math
import numpy as np

from constant import POSE

class Utils():
    
    def __init__(self):
        ### Initial Variable ###
        # Position
        self.PositionX = self.PositionY = np.empty((0,33), float)
        self.shoulderDisplacement = np.array([])
        self.elbowDisplacement = np.array([])
        # Angle
        self.shoulderAngle = np.array([])
        self.elbowAngle = np.array([])
        self.motionAngle = np.array([])
        # Angular Velocity
        self.angularVelocity = np.array([])
        self.angularAcceleration = np.array([])

    ############### Utils Function ###############
    def splitFrames(self, m, n):
        # Normalize Data to 100 Frames
        quotient = int(m / n)
        remainder = m % n

        if remainder > 0:
            return [quotient] * (n - remainder) + [quotient + 1] * remainder
        else:
            return [quotient] * n

    ############### Position ###############
    def getPosition(self, results):
        # Temporary Varible to Store Landmark x & y
        X = Y = np.array([])

        for _, landmarks in enumerate(results.pose_landmarks.landmark):
            X = np.append(X, landmarks.x)
            Y = np.append(Y, landmarks.y)

        self.PositionX = np.append(self.PositionX, [X], axis=0)
        self.PositionY = np.append(self.PositionY, [Y], axis=0)

    def getStartPosition(self, position, mType='max'):
        # Find the Lowest Position in First 2 Seconds.
        if mType == 'min': # For 'Internal Rotation' & 'Horizontal Abduction'
            lowFrame = np.argmin(position, axis=0)
            self.Start = position.min()
        elif mType == 'max':
            lowFrame = np.argmax(position, axis=0)
            self.Start = position.max()
        return lowFrame
    
    def getEndPosition(self, X, Y, aXis='Y', mType='min'):
        # Check calculated axis #
        if aXis == 'X':
            ax = X
        elif aXis == 'Y':
            ax = Y
        # Absolutely Highest Position #
        if mType == 'min': # For 'Internal Rotation' & 'Horizontal Abduction'
            absHighFrame = np.argmin(ax, axis=0)
            self.End = ax.min()
        elif mType == 'max':
            absHighFrame = np.argmax(ax, axis=0)
            self.End = ax.max()
        print(f"Abs Frame: {absHighFrame+30}")
        
        # Stable Highest Position #
        distList = np.array([])
        # for num in range(absHighFrame-30, len(ax)-15):
        inLoopCount = 0
        for num in range(len(ax)-15):
            distance = 100 # Set extreme value that out of end point range
            # print(f"{num} frame --> start: {self.Start},end: {self.End}, now: {ax[num]}")
            if abs(ax[num] - self.Start) >= abs(self.End - self.Start)*0.90:
                inLoopCount += 1
                distance = 0
                for i in range(15):
                    # before 15 frames
                    before = math.sqrt((X[num]-X[num-i])**2 + (Y[num]-Y[num-i])**2)
                    # after 15 frames
                    after = math.sqrt((X[num]-X[num+i])**2 + (Y[num]-Y[num+i])**2)
                    # Sum Before & After Distance
                    distance += before + after
            distList = np.append(distList, distance)
        if inLoopCount < 10:
            stbHighFrame = absHighFrame + 30
        else:
            stbHighFrame = int(np.argmin(distList)) + 30

        return stbHighFrame
    
    def landmarkDisplacement(self, start, X, Y, shoulder, elbow=''):
        self.shoulderDisplacement = np.append(self.shoulderDisplacement, 
                                     math.sqrt((self.PositionX[start][shoulder]-X[shoulder])**2 +
                                               (self.PositionY[start][shoulder]-Y[shoulder])**2))
        if elbow != '':
            self.elbowDisplacement = np.append(self.elbowDisplacement,
                                      math.sqrt((self.PositionX[start][elbow]-X[elbow])**2 +
                                                (self.PositionY[start][elbow]-Y[elbow])**2))
        
    ############### Angle ###############
    def calculateAngle(self, point1, point2, point3):
        # Method of Calculate The Angle of Point1 - "Point2" - Point3
        Point21 = point1 - point2
        Point23 = point3 - point2

        cosAngle = np.dot(Point21, Point23) / (np.linalg.norm(Point21) * np.linalg.norm(Point23))
        angle = np.arccos(cosAngle)

        return np.degrees(angle)
    
    def getVector(self, point1, point2):
        vec = [point2[0]-point1[0], point2[1]-point1[1]]
        return vec

    def getAngle(self, side, X, Y):
        # Send "Shoulder & Elbow" Landmarks to Calculate Method
        hip = np.array([X[POSE[side+'hip']], Y[POSE[side+'hip']]])
        shoulder = np.array([X[POSE[side+'shoulder']], Y[POSE[side+'shoulder']]])
        elbow = np.array([X[POSE[side+'elbow']], Y[POSE[side+'elbow']]])
        wrist = np.array([X[POSE[side+'wrist']], Y[POSE[side+'wrist']]])

        self.shoulderAngle = np.append(self.shoulderAngle, self.calculateAngle(hip, shoulder, elbow))
        self.elbowAngle = np.append(self.elbowAngle, self.calculateAngle(shoulder, elbow, wrist))

    def getHorzVertAngle(self, startVec, endVec, angleReverse):
        vecDot = startVec[0]*endVec[0] + startVec[1]*endVec[1]
        vecDist = (math.sqrt(startVec[0]**2 + startVec[1]**2)) * (math.sqrt(endVec[0]**2 + endVec[1]**2))
       
        cos = vecDot/vecDist
        Theta = math.degrees(math.acos(cos))
        # Hand is front of shoulder
        if angleReverse == True:
            Theta = -Theta

        self.motionAngle = np.append(self.motionAngle, Theta)
    
    def getHorzAngle(self, startShoulder, startElbow, endShoulder, endElbow, type = 'abd'):
        startLen = math.sqrt((startShoulder[0]-startElbow[0])**2 + (startShoulder[1]-startElbow[1])**2)
        endLen = math.sqrt((endShoulder[0]-endElbow[0])**2 + (endShoulder[1]-endElbow[1])**2)
        cos = endLen/startLen
        if cos>1: cos =1
        
        Theta = math.degrees(math.acos(cos)) # cos value to degree
        
        if type == 'add':
            Theta = 180 - Theta

        self.motionAngle = np.append(self.motionAngle, Theta)
    
    ############### Angular Velocity ###############
    def getAngularVelocity(self, previosAngle, currentAngle, time, count):

            angleVariation = currentAngle - previosAngle
            angularVelocity = angleVariation/time

            self.angularVelocity = np.append(self.angularVelocity, angularVelocity)

            # Calculate Angular Acceleration
            if count == 0:
                angularAcceleration = angularVelocity/time
                self.angularAcceleration = np.append(self.angularAcceleration, angularAcceleration)
            else:

                angularVelocityVariation = angularVelocity - self.angularVelocity[int(count)-1]
                angularAcceleration = angularVelocityVariation/time
                self.angularAcceleration = np.append(self.angularAcceleration, angularAcceleration)

    ############### Length ###############
    def startLength(self, X, Y, p1, p2):

        startLength = 0
        for i in range(0, 30):
            armLength = math.sqrt((X[i, p1]-X[i, p2])**2 + (Y[i, p1]-Y[i, p2])**2)
            startLength += armLength
            startLength = startLength/30

        return startLength
    
    def getLength(self, start, X, Y, p1, p2):

        armLength = math.sqrt((X[p1]-X[p2])**2 + (Y[p1]-Y[p2])**2)
        relateLength = armLength/start

        self.length = np.append(self.length, relateLength)