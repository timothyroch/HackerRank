"""

Fred's Battery Prediction – Linear Regression Implementation

This script is my solution to a regression problem where the goal is to predict how long Fred's laptop 
battery will last based on how long it was charged. The data comes from a file called "trainingdata.txt" 
that contains 100 pairs of (charge_time, battery_life) values.

Approach:
- I implemented simple linear regression manually using the least squares method.
- I read and filter the training data, keeping only valid points where both values are non-negative and
  battery life is strictly less than 8.0 hours (since the problem states the battery can’t last more than 8).
- I compute the slope (m) and intercept (b) of the regression line y = mx + b based on the cleaned dataset.
- For a given input (time charged), I use the model to predict the battery duration.
- The final prediction is capped at 8.0 hours and rounded to two decimal places.

-------------------------------------------------------------------------------------------------------------

Problem Description:

Fred is a very predictable man. For instance, when he uses his laptop, all he does is watch TV shows.
He keeps on watching TV shows until his battery dies. Also, he is a very meticulous man, i.e. he pays
great attention to minute details. He has been keeping logs of every time he charged his laptop, which
includes how long he charged his laptop for and after that how long was he able to watch the TV. Now,
Fred wants to use this log to predict how long will he be able to watch TV for when he starts so that
he can plan his activities after watching his TV shows accordingly.

Challenge:

You are given access to Fred’s laptop charging log by reading from the file “trainingdata.txt”. The 
training data file will consist of 100 lines, each with 2 comma-separated numbers.

The first number denotes the amount of time the laptop was charged.
The second number denotes the amount of time the battery lasted.
The training data file can be downloaded here (this will be the same training data used when your program
is run). The input for each of the test cases will consist of exactly 1 number rounded to 2 decimal places.
For each input, output 1 number: the amount of time you predict his battery will last.

"""

if __name__ == '__main__':
    timeCharged = float(input().strip())
    maxBattery = 8 # specified in the instructions - battery cannot last more than 8 hours

    chargedList = []
    lastedList = []
    # Read and parse the training data
    with open("trainingdata.txt") as data:
        for line in data:
            x = line.strip().split(',')
            if len(x) != 2:
                continue
            else:
                try:
                    charged = float(x[0])
                    lasted = float(x[1])
                    if charged >= 0 and lasted >= 0 and lasted < 8.0:
                        chargedList.append(charged)
                        lastedList.append(lasted)
                except  ValueError:
                    continue

    length = len(chargedList)
    if length == 0:
        print("0.00")
        exit()
      
    # Compute means of both variables
    charged_mean = sum(chargedList) / length
    lasted_mean = sum(lastedList) / length

    # Compute covariance and variance for regression coefficients
    numerator = 0
    denominator = 0
    for i in range(length):
        dx = chargedList[i] - charged_mean
        dy = lastedList[i] - lasted_mean
        numerator += dx * dy
        denominator += dx * dx

    # Calculate slope (m) and intercept (b)
    m = numerator / denominator if denominator != 0 else 0
    b  = lasted_mean- m * charged_mean

    # Predict and cap the result at max battery capacity
    prediction = min(m * timeCharged + b, 8.00)
    print(f"{prediction:.2f}")
