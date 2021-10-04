#!/usr/bin/python
import pprint
import spectral_entropy


def test_0():
    spec_query = [[100, 0], [200, 80], [300, 20]]
    spec_reference = []

    all_dist = spectral_entropy.all_distance(spec_query, spec_reference, ms2_da=0.05)
    pprint.pprint(all_dist)


# Test
def test_1():
    spec_query = [[100, 0], [200, 80], [300, 20]]
    spec_reference = [[100, 30], [200, 0], [300, 70]]

    all_dist = spectral_entropy.all_distance(spec_query, spec_reference, ms2_da=0.05)
    pprint.pprint(all_dist)
    pass


def test_2():
    spec_query = [[200.0, 100]]
    spec_reference = [[201.0, 100]]

    all_dist = spectral_entropy.all_distance(spec_query, spec_reference, ms2_da=0.05)
    pprint.pprint(all_dist)
    pass


def test_3():
    spec_query = []
    spec_reference = []

    all_dist = spectral_entropy.all_distance(spec_query, spec_reference, ms2_da=0.05)
    pprint.pprint(all_dist)


def test_4():
    spec_query = [[124.0869, 0.32267],
                  [148.9238, 20.5],
                  [156.015, 0.222153],
                  [186.0342, 40],
                  [279.0911, 50],
                  [300, 10]]
    spec_reference = [[124.0869, 0.32267],
                      [148.9238, 20.5],
                      [156.011, 0.222153],
                      [279.0912, 50],
                      [289.0911, 50]]

    spec_query = [[192.1383, 999.00],
                  [136.0757, 178.22],
                  [121.0648, 82.82],
                  [150.0914, 38.46],
                  [119.0492, 35.76],
                  [91.0540, 16.58],
                  [135.0679, 14.99],
                  [57.0697, 8.09],
                  [211.0542, 5.19]]
    spec_reference = [[91.0540, 999.00],
                      [136.0757, 738.56],
                      [119.0491, 605.09],
                      [121.0648, 477.12],
                      [192.1384, 270.73],
                      [135.0679, 195.10],
                      [57.0697, 147.95],
                      [150.0914, 41.86],
                      [211.0543, 37.86]]

    all_dist = spectral_entropy.all_distance(spec_query, spec_reference, ms2_da=0.05)
    pprint.pprint(all_dist)
    pass


def test_5():
    spec_query = [[69.071, 7.917962], [86.066, 1.021589], [86.0969, 100.0]]
    spec_reference = [[41.04, 37.16], [69.07, 66.83], [86.1, 999.0]]

    all_dist = spectral_entropy.all_distance(spec_query, spec_reference, ms2_da=0.05)
    pprint.pprint(all_dist)
    pass


def test_6():
    spec_query = [[10, 0], [11, 1]]
    spec_reference = [[10, 0]]

    all_dist = spectral_entropy.all_distance(spec_query, spec_reference, ms2_da=0.05)
    pprint.pprint(all_dist)


if __name__ == '__main__':
    test_0()
    test_0()
    test_1()
    test_5()
    test_2()
    test_3()
    test_4()
