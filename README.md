# StrokesVerify
Recognize Chinese in a picture. The step are:
1. deal with the original picture. such as blur/sobel/threshold and so on.
2. use connectedComponentsWithStats to get different connect area.
3. use getValidArea method to filter invalid area which include 4 steps.
   3.1 remove areas are too small
   3.2 connect the areas which are very closer
   3.3 verify the connected areas, remove the areas which are not belong to Chinese.
   3.4 remove the areas which include more than 2 sub area
4. filter the end areas that pass the Chinese strokes width.

use method
The main method is findarea.py, use: python findarea.py 1.png, the result will display three pictures(press space key to switch), the last picture is the finally image.
