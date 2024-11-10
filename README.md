# Dynamic Clustering Visualization Tool

This Python script provides a dynamic visualization of the partitioning clustering process using the K-Means algorithm. The distance measure used is a simplistic but appropriate 2-dimensional Euclidean distance. Sum of squared errors was not integrated to create more optimal clusters.

It allows users to input data points and cluster centers (centroids) either manually or automatically and visualizes the iterative process of clustering using `matplotlib`.

It also supports viewing previous and past iterations through interactive keyboard controls.

For optimal usage, you need to have both the `main.py` and `creatingInputs.py` files in the same directory.

## Features

Key features of this project include:

- **Dynamic Visualization**: Watch the clustering process unfold in real time.
- **Interactive Graph**: Navigate through different iterations of the clustering process.
- **Manual/Automatic Data Input**: Choose to manually input data points and clusters or generate them automatically. Alternatively, have the `creatingInputs.py` file present in the same directory to enable file input using automatically created input files. If you so wish as well, follow the guide below to create your own input file.
- **Color Optimization**: Each cluster is represented with a distinct color for better visualization based on maximally distant [CIELAB](https://en.wikipedia.org/wiki/CIELAB_color_space) color values.
- **Automatic Installation**: Just run the script and all requirements will automatically be installed so you'll never have an issue with missing packages in either your main environment or in your virtual environment if you are using one.

## Usage

Run the script using Python:

```bash
python main.py
```

NOTE: The script automatically installs all required libraries and packages by creating a "requirements.txt" file with required non-built-in packages then uses pip to install them.

### Data Input

You will be prompted to choose the input method for data points and clusters.

- File Input: Load data points and clusters from a file. 

> NOTE: This option is only available if there exists a "creatingInputs.py" file in the directory as this script is automatically run to generate sample inputs that are compatible with the file input format.

- Manual Input: Input each data point and cluster center in the format x,y.
- Automatic Input: Automatically generate a specified number of data points and clusters.

### Keyboard Controls in Visualization

- Left Arrow ⬅️: Move to the previous iteration.
- Right Arrow ➡️: Move to the next iteration.

## Input File Format

There are 2 aspects to defining an input file:

- Name
- Format

### Name

The name of the file must be:

```Text
input-{}.txt
```

Where the "{}" part can be replaced with any value you wish. An example is `input-1.txt` or `input-sample-1.txt`

### Format

If you choose to load data from a file, the format should be as follows:

```text
<number_of_clusters>
<number_of_points>
<cluster_1_x> <cluster_1_y>
...
<cluster_n_x> <cluster_n_y>
<point_1_x> <point_1_y>
...
<point_m_x> <point_m_y>
```

## Dependencies

- matplotlib
- colormath
- numpy (used in generating sample input files only)
- scikit-learn (used in generating sample input files only)

These dependencies are listed in requirements.txt that will be generated automatically if it doesn't exist and these packages will be installed automatically.

## Authors

- Fatima AlZahra AlHasan 🧠 - 20-0510
- Mike Hanna ✨ - 19-0743
