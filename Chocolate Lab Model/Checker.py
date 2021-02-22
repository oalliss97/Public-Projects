import os
os.chdir('Chocolate Lab Model')

from Checker_Homeward_Trails import hrt
from Checker_Lus_Labs import luslabs

os.chdir('..')

print("\n\nChecking Homeward Trails now...")
hrt()

print("\n\nChecking Lu's Labs now...")
luslabs()