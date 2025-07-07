# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 12:31:51 2025

@author: nickr
"""

# -------------------------------------------------------
# ✅ GWpy example: plot raw strain for any LIGO event
# -------------------------------------------------------

from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
import matplotlib.pyplot as plt

# -------------------------------------------------------
# 1️⃣ Choose an event or GPS
# -------------------------------------------------------

# Example: GW150914
event = "GW170104"
gps = event_gps(event)
print(f"{event} GPS: {gps}")

# -------------------------------------------------------
# 2️⃣ Choose a detector and time window
# -------------------------------------------------------

detector = "H1"  # Hanford or "L1" for Livingston

# Example: 4 seconds before to 4 seconds after
start = gps - 4
end   = gps + 4

# -------------------------------------------------------
# 3️⃣ Load data from GWOSC
# -------------------------------------------------------

strain = TimeSeries.fetch_open_data(detector, start, end, sample_rate=16384)

print(strain)

# -------------------------------------------------------
# 4️⃣ Plot strain
# -------------------------------------------------------

plot = strain.plot()
plot.axes[0].set_title(f"{detector} strain: {event}")
plot.show()
