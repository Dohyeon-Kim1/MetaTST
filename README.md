# MetaTST

## Train

"""python
# train TST
python main.py --mode=trian --type=basic --epoch=30

# train MetaTST
python main.py --mode=train --type=meta
"""

## Test (Genneration)

"""python
# test TST
python main.py --mode=test --type=basic

# test Meta
python main.py --mode=test --type=meta
"""

## Evaluation

"""python
# evaluation TST
python main.py --mode=evaluation --type=basic

# evaluation MetaTST
python main.py --mode=evaluation --type=meta
"""