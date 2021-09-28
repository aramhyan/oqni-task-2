def sumOfAbs(arr):
  s=0
  for i in arr:
    s+=abs(i)

  return s


def meanOfAbs(arr):
  s=0
  for i in arr:
    s+=abs(i)

  return s/len(arr)


def meanOfWeightedAbs(arr):
  s=0
  for i in range(len(arr)):
    s+=(1 if 0.25*len(arr)<=i+1<=0.75*len(arr) else 0.5)*abs(arr[i])

  return s/len(arr)


def meanOfMultiWeightedAbs(arr):
  s=0
  for i in range(len(arr)):
    s+=(1 if 0.25*len(arr)<=i+1<=0.75*len(arr) else (4*i/len(arr) if i+1<0.25*len(arr) else 4*(i-len(arr))/len(arr)))*abs(arr[i])

  return s/len(arr)


def sumOfSquares(arr):
  s=0
  for i in arr:
    s+=pow(i,2)

  return s
