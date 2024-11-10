"""
***********************************************************************************
*                                                                                 *
*                      Dynamic Clustering Visualization Tool                      *
*                                                                                 *
*  Authors: Mike Hanna (19-0743) & Fatima AlZahra AlHasan (20-0510)               *
*                                                                                 *
*  Description:                                                                   *
*  This Python script provides a dynamic visualization of the partitioning        *
*  clustering process using the K-Means algorithm. The distance measure used      *
*  is a simplistic but appropriate 2-dimensional Euclidean distance. Sum of       *
*  squared errors was not integrated to create more optimal clusters.             *
*                                                                                 *
*  It allows users to input data points and cluster centers (centroids) either    *
*  manually or automatically and visualizes the iterative process of clustering   *
*  using `matplotlib`. It also supports viewing previous and past iterations      *
*  through interactive keyboard controls.                                         *
*                                                                                 *
*  For optimal usage, you need to have both the `main.py` and `creatingInputs.py` *
*  files in the same directory.                                                   *
*                                                                                 *
*  Key Features:                                                                  *
*  - Dynamic Visualization                                                        *
*  - Interactive Graph                                                            *
*  - Manual/Automatic Data Input                                                  *
*  - Color Optimization                                                           *
*  - Automatic Installation                                                       *
*                                                                                 *
***********************************************************************************
"""


import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
import math
import time
import os
import subprocess
import sys


# Global Variable Initializations
global_pointsList = []  # example Point (tuple): (x: float, y: float)
global_clustersList = []  # example Cluster: [x: float, y: float, points: list]
global_scatterPlotsList = []
global_iterationMemoryList = []  # For viewing previous iterations
global_currentIterationIndex = 0  # For viewing previous iterations
global_continueIteratingBool = True
global_generatedColorsList = []
global_chosenColorsForPlotsList = []

# Global Variable Initializations for Optimization Metrics
global_viewOptimizationMetricsBool = False
global_previousSseValue = math.inf
global_previousSsbValue = math.inf
global_previousSstValue = math.inf
global_previousSilhouetteScore = -1
global_sseScoresList = []
global_ssbScoresList = []
global_sstScoresList = []
global_silhouetteScoresList = []


# Input Generation Functions
def generateRandomPoints(n: int, loggingLevel=0):
    pointsList = []
    for _ in range(n):
        point = (random.random() * n, random.random() * n)
        pointsList.append(point)

        if loggingLevel >= 1:
            print("Generated Point:", point)
    return pointsList


def generateRandomClusters(
    n: int, minX: float, maxX: float, minY: float, maxY: float, loggingLevel=0
):
    clustersList = []

    for _ in range(n):
        cluster = [
            random.uniform(minX - 0.1 * abs(minX), maxX + 0.1 * abs(maxX)),
            random.uniform(minY - 0.1 * abs(minY), maxY + 0.1 * abs(maxY)),
            [],
        ]
        clustersList.append(cluster)

        if loggingLevel >= 1:
            print("Generated Cluster:", cluster)
    return clustersList


def inputPoints(loggingLevel=0):
    points = []
    while True:
        inp = input("> Enter point as 'x,y' or 'q' to quit: ")
        if inp.lower() == "q":
            break
        try:
            x, y = map(float, inp.split(","))
            points.append((x, y))

            if loggingLevel == 1:
                print("Added point at:", (x, y))
        except ValueError:
            print("Invalid input. Please enter in 'x,y' format.")
    return points


def inputClusters(loggingLevel=0):
    clusters = []
    while True:
        inp = input("> Enter cluster as 'x,y' or 'q' to quit: ")
        if inp.lower() == "q":
            break
        try:
            x, y = map(float, inp.split(","))
            clusters.append([x, y, []])

            if loggingLevel == 1:
                print("Added cluster at:", (x, y))
        except ValueError:
            print("Invalid input. Please enter in 'x,y' format.")
    return clusters


# Optimization Metrics
def calculateSSE(clusters):
    sse = 0
    for cluster in clusters:
        centroid = (cluster[0], cluster[1])
        for point in cluster[2]:
            sse += calculateEuclideanDistance(point, centroid) ** 2
    return sse


def calculateOverallDataMean(points):
    sumX = sum(point[0] for point in points)
    sumY = sum(point[1] for point in points)
    return sumX / len(points), sumY / len(points)


def calculateSSB(clusters, overall_mean):
    ssb = 0
    for cluster in clusters:
        n = len(cluster[2])
        centroid = (cluster[0], cluster[1])
        ssb += n * calculateEuclideanDistance(centroid, overall_mean) ** 2
    return ssb


def calculateSST(points, overall_mean):
    sst = 0
    for point in points:
        sst += calculateEuclideanDistance(point, overall_mean) ** 2
    return sst


def calculateSilhouetteScore(clusters, points):
    silhouette_scores = []
    for point in points:
        # Find the cluster this point belongs to
        own_cluster = findNearestCluster(point, clusters)
        a = sum(
            calculateEuclideanDistance(point, other_point)
            for other_point in own_cluster[2]
        ) / len(own_cluster[2])

        # Find the nearest cluster that this point does not belong to
        nearest_other_cluster = min(
            [cluster for cluster in clusters if cluster != own_cluster],
            key=lambda c: calculateEuclideanDistance(point, (c[0], c[1])),
        )
        b = sum(
            calculateEuclideanDistance(point, other_point)
            for other_point in nearest_other_cluster[2]
        ) / len(nearest_other_cluster[2])

        silhouette_score = (b - a) / max(a, b)
        silhouette_scores.append(silhouette_score)

    return sum(silhouette_scores) / len(silhouette_scores)


def generateOptimizationMetrics(loggingLevel=1):
    global global_previousSseValue, global_previousSilhouetteScore, global_previousSsbValue, global_previousSstValue
    global global_sseScoresList, global_ssbScoresList, global_sstScoresList, global_silhouetteScoresList

    # Optimization Metrics
    if global_viewOptimizationMetricsBool:
        initialTime = time.time()

        overall_mean = calculateOverallDataMean(global_pointsList)
        sse = calculateSSE(global_clustersList)
        ssb = calculateSSB(global_clustersList, overall_mean)
        sst = calculateSST(global_pointsList, overall_mean)
        silhouette_score = calculateSilhouetteScore(
            global_clustersList, global_pointsList
        )
        finishTime = time.time()

        if loggingLevel >= 1:
            print("\nOptimization Metrics for Current Iteration:")
            print(
                f"pSSE: {global_previousSseValue}, pSSB: {global_previousSsbValue}, pSST: {global_previousSstValue}, pSilhouette Score: {global_previousSilhouetteScore}"
            )
            print(
                f"SSE: {sse}, SSB: {ssb}, SST: {sst}, Silhouette Score: {silhouette_score}"
            )
            print(
                f"🔺SSE: {abs(sse - global_previousSseValue)}, 🔺SSB: {abs(ssb - global_previousSsbValue)}, 🔺SST: {abs(sst - global_previousSstValue)}, 🔺Silhouette Score: {abs(silhouette_score - global_previousSilhouetteScore)}"
            )

        if loggingLevel >= 2:
            print("Calculating Metrics took:", finishTime - initialTime, "seconds")

        # Update previous values
        global_previousSseValue = sse
        global_previousSsbValue = ssb
        global_previousSstValue = sst
        global_previousSilhouetteScore = silhouette_score

        # Add to memory
        global_sseScoresList.append(sse)
        global_ssbScoresList.append(ssb)
        global_sstScoresList.append(sst)
        global_silhouetteScoresList.append(silhouette_score)


# K-Means Algorithm Functions
def calculateClusterPosition(pointsOfCluster: list):
    sumX = 0
    sumY = 0
    numberOfPoints = len(pointsOfCluster)

    for point in pointsOfCluster:
        sumX += point[0]
        sumY += point[1]

    meanX = sumX / numberOfPoints
    meanY = sumY / numberOfPoints

    return [meanX, meanY]


def calculateEuclideanDistance(point1: tuple, point2: tuple):
    return math.sqrt(
        math.pow(point2[0] - point1[0], 2) + math.pow(point2[1] - point1[1], 2)
    )


def calculateEuclideanDistance3D(point1: tuple, point2: tuple):
    return math.sqrt(
        math.pow(point2[0] - point1[0], 2)
        + math.pow(point2[1] - point1[1], 2)
        + math.pow(point2[2] - point1[2], 2)
    )


def findNearestCluster(point: tuple, clustersList: list, loggingLevel=0):
    nearestDistance = math.inf
    nearestCluster = None

    for cluster in clustersList:
        currentDistance = calculateEuclideanDistance(point, (cluster[0], cluster[1]))

        if currentDistance < nearestDistance:
            nearestDistance = currentDistance
            nearestCluster = cluster

    # * DEBUG
    if loggingLevel == 1:
        print("For the point", point, "the nearest cluster is:")
        print(nearestCluster)

    return nearestCluster


# Check if all items have NOT changed to continue iterating
def checkClusterPositionsNotChanged(
    oldClusters: list, newClusters: list, tolerance: float = 1e-6
):
    changeChecks = [
        abs(oldClusters[i][0] - newClusters[i][0]) < tolerance
        and abs(oldClusters[i][1] - newClusters[i][1]) < tolerance
        for i in range(len(oldClusters))
    ]
    for check in changeChecks:
        if not check:
            return False
    return True


def findMostDistantPointInCluster(
    clusterToAddToIndex: int, clustersList: list, loggingLevel=0
):
    pointToRemoveIndex = 0
    clusterToRemoveFromIndex = 0
    greatestAverageDistance = 0

    for cluster in clustersList:
        if len(cluster[2]) > 1:
            distancesBetweenPointsAndCentroid = [
                calculateEuclideanDistance(point, (cluster[0], cluster[1]))
                for point in cluster[2]
            ]
            maximumDistanceOfPoints = max(distancesBetweenPointsAndCentroid)
            averageDistance = sum(distancesBetweenPointsAndCentroid) / len(
                distancesBetweenPointsAndCentroid
            )

            if averageDistance > greatestAverageDistance:
                greatestAverageDistance = averageDistance
                pointToRemoveIndex = distancesBetweenPointsAndCentroid.index(
                    maximumDistanceOfPoints
                )
                clusterToRemoveFromIndex = clustersList.index(cluster)

    poppedPoint = clustersList[clusterToRemoveFromIndex][2].pop(pointToRemoveIndex)

    clustersList[clusterToAddToIndex][2].append(poppedPoint)


def findNearestPointToCluster(cluster, pointsList, loggingLevel=0):
    nearestDistance = math.inf
    nearestPoint = pointsList[0]

    for point in pointsList:
        currentDistance = calculateEuclideanDistance(point, (cluster[0], cluster[1]))

        if currentDistance < nearestDistance:
            nearestDistance = currentDistance
            nearestPoint = point

    if loggingLevel >= 1:
        print(
            "Nearest Point to Cluster", (cluster[0], cluster[1]), "is the Point", point
        )

    return nearestPoint


def addPointToNearestCluster(point, clusterList, loggingLevel=0):
    nearestCluster = findNearestCluster(point, clusterList)
    for cluster in clusterList:
        if cluster[:2] == nearestCluster[:2]:
            cluster[2].append(point)
            break

    if loggingLevel == 1:
        print(
            "Assigned point",
            point,
            "to cluster",
            nearestCluster[:2],
        )


# Utility Functions
def RGB_to_LAB(rgb):
    return convert_color(sRGBColor(*rgb, is_upscaled=True), LabColor)


def generateMaximallyDistantInitialColors(n):
    # Generate initial set of random colors
    colors_rgb = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(n + 1)]
    colors_lab = [RGB_to_LAB(rgb) for rgb in colors_rgb]

    # No need to maximize distance between colors if only 1 color is needed, just get a random one
    if n < 2:
        final_colors = [
            color.get_rgb_hex()
            for color in map(lambda lab: convert_color(lab, sRGBColor), colors_lab)
        ]
        return final_colors

    # Optimize the colors to maximize the minimum distance between any two colors
    for _ in range(1000):
        for j in range(n + 1):
            current_color = colors_lab[j]
            others = colors_lab[:j] + colors_lab[j + 1 :]
            min_distance = min(
                calculateEuclideanDistance3D(
                    current_color.get_value_tuple(), c.get_value_tuple()
                )
                for c in others
            )
            # Try to find a new color that increases the minimum distance
            new_color = tuple(random.randint(0, 255) for _ in range(3))
            new_color_lab = RGB_to_LAB(new_color)
            new_min_distance = min(
                calculateEuclideanDistance3D(
                    new_color_lab.get_value_tuple(), c.get_value_tuple()
                )
                for c in others
            )
            if new_min_distance > min_distance:
                colors_lab[j] = new_color_lab

    # Convert LAB colors back to RGB
    final_colors = [
        color.get_rgb_hex()
        for color in map(lambda lab: convert_color(lab, sRGBColor), colors_lab)
    ]
    return final_colors


def chooseRandomColor():
    global global_generatedColorsList
    global global_chosenColorsForPlotsList

    colorChoice = global_generatedColorsList[
        math.floor(random.random() * len(global_generatedColorsList))
    ]

    while colorChoice in global_chosenColorsForPlotsList:
        colorChoice = global_generatedColorsList[
            math.floor(random.random() * len(global_generatedColorsList))
        ]
    return colorChoice


def adjustHexColorShade(hex_color: str, factor=0.75, loggingLevel=0):
    # Ensure the hex color starts with '#'
    if not hex_color.startswith("#"):
        hex_color = "#" + hex_color

    # Extract the red, green, and blue components
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)

    # Multiply each component by the factor and ensure it's within the range 0-255
    r = min(255, max(0, int(r * factor)))
    g = min(255, max(0, int(g * factor)))
    b = min(255, max(0, int(b * factor)))

    if loggingLevel >= 1:
        print("New Color:", f"#{r:02x}{g:02x}{b:02x}")

    # Convert back to hexadecimal and return
    return f"#{r:02x}{g:02x}{b:02x}"


# Graph Updater Function that fires every interval to animate the graph
def updateGraph(frame):
    global global_clustersList
    global global_continueIteratingBool
    global global_iterationMemoryList
    global global_currentIterationIndex

    stringToPrint = "Staring Iteration " + str(len(global_iterationMemoryList) + 1)
    print()
    print("#" * (len(stringToPrint) + 6))
    print("##", stringToPrint, "##")
    print("#" * (len(stringToPrint) + 6))

    tempClusters = []
    for cluster in global_clustersList:
        newPosition = calculateClusterPosition(cluster[2])
        tempClusters.append([newPosition[0], newPosition[1], []])

        print(
            "\nChanging Cluster Position of Cluster",
            global_clustersList.index(cluster) + 1,
        )
        print("Old Position      :", (cluster[0], cluster[1]))
        print("New Position      :", (newPosition[0], newPosition[1]))
        print(
            "Change in Position:",
            (abs(cluster[0] - newPosition[0]), abs(cluster[1] - newPosition[1])),
        )
        # print("Timestamp:", round(time.time() * 1000))

    if checkClusterPositionsNotChanged(global_clustersList, tempClusters):
        global_continueIteratingBool = False

        iterationData = [
            (
                plot.get_offsets(),
                plot.get_facecolor(),
                plot.get_sizes(),
                plot.get_alpha(),
            )
            for plot in global_scatterPlotsList
        ]
        global_iterationMemoryList.append(iterationData)

        global_currentIterationIndex = len(global_iterationMemoryList) - 1

        # Generating and Storing Optimization Metrics
        generateOptimizationMetrics(loggingLevel=2)

        # End updates
        stringToPrint = (
            "Graph has converged. There were Iteration "
            + str(len(global_iterationMemoryList))
            + " iterations."
        )
        print()
        print("#" * (len(stringToPrint) + 6))
        print("##", stringToPrint, "##")
        print("#" * (len(stringToPrint) + 6))

        print(
            "\nYou may now use the left '⬅️' and right '➡️' arrow keys to view different iterations."
        )
        return

    for point in global_pointsList:
        addPointToNearestCluster(point, tempClusters)

    global_clustersList = tempClusters[:]

    for i, cluster in enumerate(tempClusters):
        # Update cluster points scatter plot
        global_scatterPlotsList[2 * i + 1].set_offsets([[cluster[0], cluster[1]]])

        # Update cluster center scatter plot
        clusterPointsXValues = [item[0] for item in cluster[2]]
        clusterPointsYValues = [item[1] for item in cluster[2]]
        global_scatterPlotsList[2 * i].set_offsets(
            list(zip(clusterPointsXValues, clusterPointsYValues))
        )

    # Optimization Metrics
    generateOptimizationMetrics(loggingLevel=2)

    # Saving into memory for viewing previous interactions
    iterationData = [
        (plot.get_offsets(), plot.get_facecolor(), plot.get_sizes(), plot.get_alpha())
        for plot in global_scatterPlotsList
    ]
    global_iterationMemoryList.append(iterationData)

    return (global_scatterPlotsList,)


# Enables stopping the animation once the graph converges
def frameGenerator():
    global global_continueIteratingBool
    frame = 0
    while global_continueIteratingBool:
        yield frame
        frame += 1


def on_key(event, ax):
    global global_currentIterationIndex
    oldIterationIndex = global_currentIterationIndex
    if event.key in ["left", "right"]:
        if (
            event.key == "right"
            and global_currentIterationIndex < len(global_iterationMemoryList) - 1
        ):
            global_currentIterationIndex += 1
            print("\nPressed Right ➡️")
        elif event.key == "left" and global_currentIterationIndex > 0:
            global_currentIterationIndex -= 1
            print("\nPressed Left ⬅️")
        elif (
            global_currentIterationIndex == 0
            or global_currentIterationIndex == len(global_iterationMemoryList) - 1
        ):
            return

        print("Previous Iteration Number:", oldIterationIndex + 1)
        print("Current Iteration Number :", global_currentIterationIndex + 1)
        print("Can go up to Iteration   :", len(global_iterationMemoryList))

        updateDisplayUsingHistoricalData(global_currentIterationIndex, ax)


def updateDisplayUsingHistoricalData(index, ax):
    global global_sseScoresList
    global global_ssbScoresList
    global global_sstScoresList
    global global_silhouetteScoresList
    global global_currentIterationIndex

    # Clear the current plots
    ax.clear()

    # Recreate the plots using the stored data
    iterationData = global_iterationMemoryList[index]
    scatterPlotsList = []
    for data in iterationData:
        offsets, color, size, alpha = data
        scatterPlot = ax.scatter(
            offsets[:, 0], offsets[:, 1], color=color, s=size, alpha=alpha
        )
        scatterPlotsList.append(scatterPlot)

    if global_viewOptimizationMetricsBool:
        # if global_currentIterationIndex > 0:
        sse = global_sseScoresList[global_currentIterationIndex - 1]
        ssb = global_ssbScoresList[global_currentIterationIndex - 1]
        sst = global_sstScoresList[global_currentIterationIndex - 1]
        silhouette_score = global_silhouetteScoresList[global_currentIterationIndex - 1]
        print(
            f"SSE: {sse}, SSB: {ssb}, SST: {sst}, Silhouette Score: {silhouette_score}"
        )

    # Redraw the axes and labels as needed
    ax.set_xlabel("Y Position")
    ax.set_ylabel("X Position")
    ax.set_title("Clustering")
    plt.draw()


# File input
def loadDataFromInputFile(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
        n_clusters = int(lines[0].strip())
        n_points = int(lines[1].strip())
        clusters = [
            list(map(float, line.strip().split())) for line in lines[2 : 2 + n_clusters]
        ]
        for cluster in clusters:
            cluster.append([])
        points = [
            tuple(map(float, line.strip().split()))
            for line in lines[2 : 2 + n_clusters + n_points]
        ]
    return clusters, points


def listInputFilesInDirectory():
    files = [
        f for f in os.listdir(".") if f.startswith("input-") and f.endswith(".txt")
    ]
    return files


# Driver Code
def main():
    global global_scatterPlotsList
    global global_pointsList
    global global_clustersList
    global global_generatedColorsList
    global global_iterationMemoryList
    global global_viewOptimizationMetricsBool

    input_choice = input(
        "> Choose input method - Manual/Automatic (m/a) or File (f): "
    ).lower()
    if input_choice == "f":
        files = listInputFilesInDirectory()
        if files:
            print("Available input files:")
            for i, file in enumerate(files):
                print(f"{i + 1}. {file}")
            file_index = int(input("> Select a file number: ")) - 1
            if 0 <= file_index < len(files):
                global_clustersList, global_pointsList = loadDataFromInputFile(
                    files[file_index]
                )
            else:
                print("Invalid file selection. Exiting.")
                return
        else:
            print("No input files found. Exiting.")
            return
    elif input_choice in ["m", "a", "m/a"]:
        # Input points
        choice = input(
            "> Do you want to input points manually or automatically? (m/a): "
        )
        if choice.lower() == "m":
            global_pointsList = inputPoints()
        elif choice.lower() == "a":
            n = int(input("> How many points do you want to generate? "))
            global_pointsList = generateRandomPoints(n)
        else:
            print("Invalid choice. Exiting.")
            return

        # Input clusters
        choice = input(
            "> Do you want to input clusters manually or automatically? (m/a): "
        )
        if global_pointsList:
            minX = min(p[0] for p in global_pointsList)
            maxX = max(p[0] for p in global_pointsList)
            minY = min(p[1] for p in global_pointsList)
            maxY = max(p[1] for p in global_pointsList)

            if choice.lower() == "m":
                global_clustersList = inputClusters()
            elif choice.lower() == "a":
                z = int(input("> How many clusters do you want to generate? "))
                global_clustersList = generateRandomClusters(z, minX, maxX, minY, maxY)
            else:
                print("Invalid choice. Exiting.")
                return
        else:
            print("No points were input. Exiting.")
            return

    # Choosing to view optimization metrics or not
    choice = input(
        "> Do you want to view optimization metric (SSE, SSB, SSt, Silhouette Coefficient) [this will cause a slow-down]? (y/n): "
    )
    if choice.lower() == "y":
        global_viewOptimizationMetricsBool = True
    elif choice.lower() == "n":
        pass
    else:
        print("Invalid choice. Exiting.")
        return

    # Rest of code
    global_generatedColorsList = generateMaximallyDistantInitialColors(
        len(global_clustersList)
    )

    for point in global_pointsList:
        addPointToNearestCluster(point, global_clustersList, 1)

    for cluster in global_clustersList:
        if len(cluster[2]) == 0:
            findMostDistantPointInCluster(
                global_clustersList.index(cluster), global_clustersList
            )

    # Optimization Metrics
    generateOptimizationMetrics(loggingLevel=2)

    # Generate Scatter Plots for each Cluster Set
    fig, ax = plt.subplots()
    for cluster in global_clustersList:
        # Generating Colors of Points and Darker Shade for Cluster
        chosenColor = chooseRandomColor()
        global_chosenColorsForPlotsList.append(chosenColor)
        clusterColor = adjustHexColorShade(chosenColor)

        # Generate the plot for the cluster's points
        clusterPointsXValues = [item[0] for item in cluster[2]]
        clusterPointsYValues = [item[1] for item in cluster[2]]

        global_scatterPlotsList.append(
            ax.scatter(
                clusterPointsXValues,
                clusterPointsYValues,
                color=chosenColor,
            )
        )

        # Generate the plot for the cluster center alone
        global_scatterPlotsList.append(
            ax.scatter([cluster[0]], [cluster[1]], color=clusterColor, s=200, alpha=0.5)
        )

    # Enables remembering previous iterations
    iterationData = [
        (plot.get_offsets(), plot.get_facecolor(), plot.get_sizes(), plot.get_alpha())
        for plot in global_scatterPlotsList
    ]
    global_iterationMemoryList.append(iterationData)

    plt.xlabel("Y Position")
    plt.ylabel("X Position")
    plt.title("Clustering")

    animation = FuncAnimation(
        fig,
        updateGraph,
        frames=frameGenerator,
        interval=1000,
        repeat=False,
        cache_frame_data=False,
    )

    fig.canvas.mpl_connect("key_press_event", lambda event: on_key(event, ax))
    plt.show()


def installPackageRequirements():
    requirements = [
        "matplotlib",
        "colormath",
        "numpy",
        "scikit-learn",
    ]

    with open("requirements.txt", "w") as f:
        for req in requirements:
            f.write(req + "\n")

    print("Installing required libraries:")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    )
    print()


def createSampleInputFiles():
    subprocess.check_call([sys.executable, "creatingInputs.py"])


if __name__ == "__main__":
    expectedInputFiles = [
        "input-1.txt",
        "input-2.txt",
        "input-3.txt",
        "input-4.txt",
        "input-5.txt",
        "input-6.txt",
        "input-7.txt",
    ]
    installPackageRequirements()

    if (
        os.path.exists("creatingInputs.py")
        and len(
            [
                filename
                for filename in expectedInputFiles
                if not os.path.exists(filename)
            ]
        )
        > 0
    ):
        createSampleInputFiles()

    main()
