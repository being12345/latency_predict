# checkList for Model

## network alg
<input type="checkbox">  network init? </br>

## loss
<input type="checkbox">  loss input, target, output? </br> 
1. no state 
    1. X, edge_index, edge_weight

## optimizer
<input type="checkbox">  what optimizer? Adam </br>
not change 

## training_step
<input type="checkbox">  How to decomposition batch? </br>
<input type="checkbox">  forward and loss? </br>
<input type="checkbox">  update log? </br>
<input type="checkbox">  return loss(must included, maybe state)? </br>
1.  x
2. loss L2 metrics: L1
3. ok
4. only loss

## validation_step
<input type="checkbox">  How to decomposition batch? </br>
<input type="checkbox">  choose metrics ? </br>
<input type="checkbox">  log? </br>
1. L1 MS

## test_step
<input type="checkbox">  any special? </br>

