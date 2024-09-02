# PlaneIntersectionFinder

## Description

**PlaneIntersectionFinder** is a Python class designed to compute the line of intersection between two planes in 3D space. The class takes in two plane equations, computes the intersection line if it exists, and returns the line's equation. Additionally, the class provides an interactive 3D visualization that shows the two planes and their intersection line.

## Features

- **Intersection Line Calculation**: The algorithm calculates the line of intersection between two planes by finding the cross product of their normal vectors and solving the system of equations formed by the planes.
- **Equation Formatting**: The intersection line is returned in a readable format, consistent with the input format of the plane equations.
- **Interactive Visualization**: The class generates an interactive 3D plot using Plotly, allowing you to visualize the two planes and the intersection line.

## How It Works

### Initialization:
You initialize the `PlaneIntersectionFinder` class with two plane equations. Each plane equation should be in the format `"ax + by + cz + d = 0"`, where `a`, `b`, `c`, and `d` are the coefficients.

### Finding Intersection Line:
The `find_intersection_line` method calculates the direction vector of the intersection line by taking the cross product of the normal vectors of the two planes. It then finds a point on the line by solving the system of linear equations formed by the two plane equations. The method returns the equation of the intersection line in the same format as the input plane equations.

### Visualization:
The `visualize` method creates an interactive 3D plot that shows the two planes and the intersection line. This visualization allows for an in-depth exploration of how the planes intersect in 3D space.
