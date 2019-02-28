"""
*** Perceptrons ***

Using a step function

Refreshes some of the content from term 1 in Perceptrons.

*** Trick to make a line go closer to a point ***

Modify the eq of the line using the coords of the point, plus bias unit

3x1 + 4x2 - 10 = 0, point (4,5)

3 4 -10
subtract 4 5 1
= -1 -1 -11

new line moved too much

so add a learning rate, and compare again

3 4 -10
subtract 4*0.1 5*0.1 1*0.1
= 2.6 3.5 -10.1

subtract if line is below the point
add if the line is above the point


"""