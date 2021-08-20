from raytracing import *

path = ImagingPath()
path.fanAngle = 0.0
path.fanNumber = 5
path.rayNumber = 100
path.objectHeight = 2.5
path.append(Space(d=40));
path.append(Lens(f=40, diameter=25));
path.append(Space(d=115));
path.append(Lens(f=75, diameter=25));
path.append(Space(d=175));
path.append(Lens(f=100, diameter=25));
path.append(Space(d=250));
path.append(Lens(f=150, diameter=25));
path.append(Space(d=150));

path.display();