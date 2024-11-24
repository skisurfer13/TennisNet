# Using equation of line (2 point formula)
def line_equation(x1, y1, x2, y2, x, y):
    try:
        val = y - y1 - ((y2 - y1)/(x2 - x1))*(x - x1)
    except:
        # Avoiding zero division error
        val = y - y1 - ((y2 - y1)/(x2 - x1 + 1e-5))*(x - x1)
    return val

def in_out_checker(kps_court, ball_court_coord):
    x_ball = ball_court_coord[0]
    y_ball = ball_court_coord[1]
    
    # Equation of line by points 4 and 5
    val_1 = line_equation(kps_court[4][0, 0], kps_court[4][0, 1],
                  kps_court[5][0, 0], kps_court[5][0, 1],
                  x_ball, y_ball)

    # Equation of line by points 6 and 7
    val_2 = line_equation(kps_court[6][0, 0], kps_court[6][0, 1],
                  kps_court[7][0, 0], kps_court[7][0, 1],
                  x_ball, y_ball)

    # Equation of line by points 4 and 6
    val_3 = line_equation(kps_court[4][0, 0], kps_court[4][0, 1],
                  kps_court[6][0, 0], kps_court[6][0, 1],
                  x_ball, y_ball)

    # Equation of line by points 5 and 7
    val_4 = line_equation(kps_court[5][0, 0], kps_court[5][0, 1],
                  kps_court[7][0, 0], kps_court[7][0, 1],
                  x_ball, y_ball)
    
    if val_1 < 0 or val_2 < 0 or val_3 < 0:
        return 1
    elif val_4 > 0:
        return 1
    return 0