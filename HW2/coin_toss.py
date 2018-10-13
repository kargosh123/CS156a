import numpy

def coinToss(numRuns, numCoins, numFlips):
	vone = []
	vrand = []
	vmin = []
	for i in range(0, numRuns):
		coinhead = []
		cointail = []
		for j in range(0, numCoins):
			temp = flipCoin(numFlips)
			coinhead.append(temp[0])
			cointail.append(temp[1])
		cone = coinhead[0]
		crand = coinhead[numpy.random.randint(0, numCoins)]
		coinhead.sort()
		cmin = coinhead[0]
		vone.append(cone/10)
		vrand.append(crand/10)
		vmin.append(cmin/10)
	return (numpy.average(vone), numpy.average(vrand), numpy.average(vmin))


def flipCoin(numFlips):
	heads = 0
	tails = 0
	for i in range(0, numFlips):
		temp = numpy.random.random_integers(0, 1)
		if temp == 0:
			heads += 1
		if temp == 1:
			tails += 1
	return (heads, tails)

print(coinToss(100000, 1000, 10)[:])