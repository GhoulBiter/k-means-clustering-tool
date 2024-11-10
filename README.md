# Dynamic Clustering Visualization Tool

This Python script dynamically visualizes the partitioning clustering process using the K-Means algorithm, providing an educational foundation for those learning or debugging clustering techniques. Using a straightforward 2-dimensional Euclidean distance measure, the script demonstrates the iterative clustering process in real-time through `matplotlib` visualizations. Notably, the project does not integrate sum of squared errors as it was not a requirement for this tool, keeping the focus on core clustering mechanics.

Users can input data points and cluster centers (centroids) either manually or automatically, with the option to view clustering iterations through interactive keyboard controls. This tool is ideal for those entering the domain of unsupervised learning, as K-Means is one of the fundamental algorithms in this field.

To ensure optimal usage, place both `main.py` and `creatingInputs.py` in the same directory.

## Features

Key features of this project include:

- **Dynamic Visualization**: Watch the clustering process unfold in real time.
- **Interactive Graph**: Navigate through different iterations of the clustering process.
- **Manual/Automatic Data Input**: Choose to manually input data points and clusters or generate them automatically. Alternatively, have the `creatingInputs.py` file present in the same directory to enable file input using automatically created input files. If you so wish as well, follow the guide below to create your own input file.
- **Color Optimization**: Each cluster is represented with a distinct color for better visualization based on maximally distant [CIELAB](https://en.wikipedia.org/wiki/CIELAB_color_space) color values.
- **Interactive Visualization with Keyboard Controls**: Navigate through iterations of clustering using keyboard arrows
- **Automatic Installation**: Just run the script and all requirements will automatically be installed so you'll never have an issue with missing packages in either your main environment or in your virtual environment if you are using one.

## Installation and Usage

Run the script using Python:

```bash
python main.py
```

> The script handles dependency installation automatically by generating a `requirements.txt` file if it doesn’t already exist and using pip to install any required packages. This process ensures compatibility in both global and virtual environments. After installing dependencies, the script also runs `creatingInputs.py` to generate sample input files for experimentation.

### Data Input

You will be prompted to choose the input method for data points and clusters.

- File Input: Load data points and clusters from a file. This option is available if the "creatingInputs.py" file is present in the directory, as it generates compatible sample inputs automatically.
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

Where the "{}" part can be replaced with any value you wish. A few examples are `input-1.txt` or `input-sample-1.txt`

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

The tool requires the following packages, which are automatically installed if missing:

- `matplotlib`
- `colormath`
- `numpy` (used in generating sample input files only)
- `scikit-learn` (used in generating sample input files only)

## Authors

- Fatima AlZahra AlHasan 🧠 - 20-0510
- Mike Hanna ✨ - 19-0743
