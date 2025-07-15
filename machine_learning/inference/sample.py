import pandas as pd

# Define camp boundaries as (xmin, xmax, ymin, ymax)
camps = {
    "Blue Buff": (3000, 4000, 8000, 9000),
    "Gromp": (3500, 4500, 7500, 8200),
    "Wolves": (4200, 4800, 7300, 7800),
    "Raptors": (4800, 5300, 7000, 7600),
    "Red Buff": (6000, 6800, 6000, 6600),
    "Krugs": (6300, 7000, 5400, 6000),
    # Add more camps as needed...
}


def line_intersects_rect(x1, y1, x2, y2, rect):
    xmin, xmax, ymin, ymax = rect

    # Check if either endpoint is inside rectangle
    if (xmin <= x1 <= xmax and ymin <= y1 <= ymax) or (
        xmin <= x2 <= xmax and ymin <= y2 <= ymax
    ):
        return True

    # Define function for line intersection with vertical/horizontal lines
    def line_intersect(p1, p2, q1, q2):
        # Check if line segments p1-p2 and q1-q2 intersect
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (
            ccw(p1, p2, q1) != ccw(p1, p2, q2)
        )

    line = ((x1, y1), (x2, y2))

    # Rectangle edges
    edges = [
        ((xmin, ymin), (xmin, ymax)),
        ((xmin, ymax), (xmax, ymax)),
        ((xmax, ymax), (xmax, ymin)),
        ((xmax, ymin), (xmin, ymin)),
    ]

    # Check intersection with any edge
    for edge in edges:
        if line_intersect(line[0], line[1], edge[0], edge[1]):
            return True

    return False


def camps_passed_through(x_prev, y_prev, x_curr, y_curr, camps_dict):
    passed_camps = []
    for camp_name, bounds in camps_dict.items():
        if line_intersects_rect(x_prev, y_prev, x_curr, y_curr, bounds):
            passed_camps.append(camp_name)
    return passed_camps


# Example DataFrame with previous and current positions:
data = {
    "X_prev": [3100, 4600, 6500],
    "Y_prev": [8100, 7200, 5900],
    "X_curr": [3600, 4900, 6700],
    "Y_curr": [8200, 7400, 5800],
}

df = pd.DataFrame(data)

# Apply the function to get camps passed through each timestep
df["camps_cleared"] = df.apply(
    lambda row: camps_passed_through(
        row["X_prev"], row["Y_prev"], row["X_curr"], row["Y_curr"], camps
    ),
    axis=1,
)

print(df[["X_prev", "Y_prev", "X_curr", "Y_curr", "camps_cleared"]])
