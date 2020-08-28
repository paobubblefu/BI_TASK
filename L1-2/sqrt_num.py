
def sqrt_num(x:float):

    if x < 0:
        return False

    else:
        threshold = 0.0000000000001

        low = 0
        high = x 
        mid = (low + high) / 2
        while high - low > threshold:
            if mid * mid > 10:
                high = mid
            else:
                low = mid
            mid = (high + low) / 2
        return mid

a= sqrt_num(10)
print(a)