
(*
 
 Implementation of Artificial Neural Network
 
  Features:
   - validation: kfold cross validation
   - optimization: gradient descent

  Arhitecture:
   - n Inputs
   - n HiddenNeurons
   - n Outputs
   - n x 2 Hidden Layers

   * This network only uses two hidden layer,
     to avoid diminishing gradient problem.

*)

open System.IO;

/// Matrix Transpose.
let transpose xss =
 let rec f xss acc =
  match xss with
  | [] -> List.empty
  | hd::_ ->
   match hd with
   | [] -> List.rev acc
   | _ ->
    f <| List.map (List.skip 1) xss <| (List.map List.head xss)::acc
 f xss List.empty

/// The Dot Product of xs and ys.
let dot xs ys =
 let f x y a = (x * y) + a
 List.foldBack2 f xs ys 0.0

/// Square of a number (x).
let square x = x * x

/// Square Distance between two scalars.
let scalarDistSquare x y a =
 (square (x - y)) + a

/// Euclidean Distance.
/// *used to calculate error for low dimension.
let distance xs ys =
 sqrt <| List.foldBack2 scalarDistSquare xs ys 0.0

/// Root Mean Square.
/// *used to calculate error for high dimension.
let rms xs ys =
 sqrt ((List.foldBack2 scalarDistSquare xs ys 0.0) / (float xs.Length))

///Shuffle List (Fisher Yates Alogrithm).
let shuffle xs =
 let f (rand: System.Random) (xs:List<'a>) =
  let rec shuffleTo (indexes: int[]) upTo =
   match upTo with
   | 0 -> indexes
   | _ ->
    let fst = rand.Next(upTo)
    let temp = indexes.[fst]
    indexes.[fst] <- indexes.[upTo]
    indexes.[upTo] <- temp
    shuffleTo indexes (upTo - 1)
  let length = xs.Length
  let indexes = [| 0 .. length - 1 |]
  let shuffled = shuffleTo indexes (length - 1)
  List.permute (fun i -> shuffled.[i]) xs
 f (System.Random()) xs

/// Retrieves the value of data on a list given the list of index.
let dataAtIndex  xs_data xs_index =
 let rec f data_xs index_xs acc =
  match index_xs with
  | [] -> List.rev acc
  | hd::tl -> (f data_xs tl ((List.item hd data_xs)::acc))
 f xs_data xs_index List.empty

/// Maps a scalar to a vector using a mapper function.
let scalarToVecOp mapper ys x = List.map (mapper x) ys

/// Maps the each elements from first list (xs) to second list (ys) using the mapper function.
let mapToSecondList mapper xs ys = List.map (scalarToVecOp mapper ys) xs

/// Scalar Vector Multiplication.
let smul c xs = List.map ((*) c) xs

/// Vector Multiplication.
let mul xs ys = List.map2 (*) xs ys

/// Vector Addition.
let add xs ys = List.map2 (+) xs ys

/// Logistic Sigmoid.
let logSigmoid x = (/) 1.0 ((+) 1.0 (exp -x))

/// Derivative of Logistic Sigmoid.
let deltaLogSigmoid x = (*) x ((-) 1.0 x)

/// Derivative of TanH i.e. sec^2h.
let deltaTanH x = (/) 1.0 <| (*) (cosh x) (cosh x)

/// Generate List of Random Elements.
let listRandElems count =
 let rec f (rand:System.Random) acc c =
  match c with
  | 0 -> acc
  | _ -> f rand <| rand.NextDouble()::acc <| (-) c 1
 f (System.Random()) List.empty count

/// Gradient. dFunc is the derivative of forward squashing function.
let gradient dFunc output target = (*) <| dFunc output <| (-) target output

/// Weighted Sum with Bias.
let weightedSum inputs weights bias = add bias <| List.map (dot inputs) weights

/// Delta or The Rate of Change.
let deltas learningRate gradients netOutputs = List.map <| smul learningRate <| mapToSecondList (*) gradients netOutputs

/// Represents a Network Layer.
type Layer = {
  Inputs: float list
  Weights: float list list
  Bias: float list
  Gradients: float list
  PrevDeltas: float list list
  BiasPrevDeltas: float list
  NetOutputs: float list
  }

/// Represents a Feed Forward Network.
type Network = {
 LearningRate: float
 Momentum: float
 Inputs: float list
 FirstHiddenLayer : Layer
 SecondHiddenLayer : Layer
 OutputLayer : Layer
 TargetOutputs: float list
 }

/// Feed Forward Network.
let feedForward net =

 let firstHiddenWeightedSum = weightedSum net.Inputs net.FirstHiddenLayer.Weights net.FirstHiddenLayer.Bias
 let firstHiddenNetOutputs = List.map tanh firstHiddenWeightedSum
 let secondHiddenWeightedSum = weightedSum firstHiddenNetOutputs net.SecondHiddenLayer.Weights net.SecondHiddenLayer.Bias
 let secondHiddenNetOutputs = List.map tanh secondHiddenWeightedSum
 let outputWeightedSum = weightedSum secondHiddenNetOutputs net.OutputLayer.Weights net.OutputLayer.Bias
 let outputs = List.map tanh outputWeightedSum
 {
  net with
   FirstHiddenLayer = {
                       net.FirstHiddenLayer with
                        Inputs = net.Inputs
                        NetOutputs = firstHiddenNetOutputs
                      }
   SecondHiddenLayer = {
                        net.SecondHiddenLayer with
                         Inputs = firstHiddenNetOutputs
                         NetOutputs = secondHiddenNetOutputs
                       }
   OutputLayer = {
                  net.OutputLayer with
                   Inputs = secondHiddenNetOutputs
                   NetOutputs = outputs
                 }
 }

/// Backpropagate at Output Layer.
let bpOutputLayer n m tOutputs (layer:Layer) =

 let grads = List.map2 (gradient deltaTanH) layer.NetOutputs tOutputs
 let bpDeltas = deltas n grads layer.Inputs
 let prevDeltasWithM = List.map (smul m) layer.PrevDeltas
 let newDeltas = List.map2 add bpDeltas prevDeltasWithM
 let weightsUpdate= List.map2 add layer.Weights newDeltas
 let biasDeltas = smul n grads
 let biasPrevDeltasWithM = smul m layer.BiasPrevDeltas
 let biasNewDeltas = add biasDeltas biasPrevDeltasWithM
 let biasUpdate = add layer.Bias biasNewDeltas
 {
  layer with
   Weights = weightsUpdate
   Bias = biasUpdate
   Gradients = grads
   PrevDeltas = newDeltas
   BiasPrevDeltas = biasNewDeltas
 }

/// Backpropagate at Hidden Layer.
let bpHiddenLayer n m layer nextLayer =

 let grads = mul (List.map deltaTanH layer.NetOutputs) (List.map (dot nextLayer.Gradients) (transpose nextLayer.Weights))
 let bpDeltas = deltas n grads layer.Inputs
 let prevDeltasWithM = List.map (smul m) layer.PrevDeltas
 let newDeltas = List.map2 add bpDeltas prevDeltasWithM
 let weightsUpdate = List.map2 add layer.Weights newDeltas
 let biasDeltas = smul n grads
 let biasPrevDeltasWithM = smul m layer.BiasPrevDeltas
 let biasNewDeltas = add biasDeltas biasPrevDeltasWithM
 let biasUpdate = add layer.Bias biasNewDeltas
 {
  layer with
   Weights = weightsUpdate
   Bias = biasUpdate
   Gradients = grads
   PrevDeltas = newDeltas
   BiasPrevDeltas = biasNewDeltas
 }

/// Backpropagate Network.
let backPropagate (net:Network) =
 let bpOutputLayer = bpOutputLayer net.LearningRate net.Momentum net.TargetOutputs net.OutputLayer
 let bpHidLayerWithHyperParams = bpHiddenLayer net.LearningRate net.Momentum
 let bpSecHidLayer = bpHidLayerWithHyperParams net.SecondHiddenLayer bpOutputLayer
 let bpFirstHidLayer = bpHidLayerWithHyperParams net.FirstHiddenLayer bpSecHidLayer
 {
  net with
   OutputLayer = bpOutputLayer
   SecondHiddenLayer = bpSecHidLayer
   FirstHiddenLayer =  bpFirstHidLayer
 }

(* Utility Functions ---------------------------------------------- *)

let vectorToString (vector:List<float>) =
 let concatCommaSep (x:float) s = x.ToString("F6") + "," + s
 List.foldBack concatCommaSep vector ""

let rec matrixToString (matrix:List<List<float>>) =
 let concatStringVector vector s = vectorToString vector + ";" + s
 List.foldBack concatStringVector matrix ""

let splitToIO net = List.splitAt net.Inputs.Length

let validate net data =
 let inputs, targets = splitToIO net data
 { net with Inputs = inputs; TargetOutputs = targets } |> feedForward

let trainOnce net data =
 let inputs, targets = splitToIO net data
 { net with Inputs = inputs; TargetOutputs = targets } |> feedForward |> backPropagate

let networkDistance network = rms network.TargetOutputs network.OutputLayer.NetOutputs

let log path data = File.AppendAllText(path, data)

let logToDataFile path filename =
   let fullfilepath = path + filename
   log fullfilepath
(* ---------------------------------------------------------------- *)

/// Train Network.
let rec train epoch kfold netAcc (data_xs:float list list) =
 match epoch with
 | 0 -> netAcc
 | _ ->
  let shuffledAllData = shuffle data_xs
  let trainSet, testSet = List.splitAt kfold shuffledAllData
  let trained = List.fold trainOnce netAcc trainSet
  let trainedRms = networkDistance trained
  let validated = List.fold validate netAcc testSet
  let validatedRms = networkDistance validated
  printfn "%f, %f" trainedRms validatedRms
  train ((-) epoch 1) kfold trained data_xs

let inputSize = 2;
let hiddenSize = 3;
let outputSize = 4;

let network = {
 LearningRate = 0.01
 Momentum = 0.5
 Inputs = List.replicate inputSize 0.0
 FirstHiddenLayer = {
                     Inputs = List.empty
                     Weights = ((*) inputSize hiddenSize) |> listRandElems |> List.chunkBySize inputSize
                     Bias = listRandElems hiddenSize
                     Gradients = List.empty
                     PrevDeltas = List.replicate hiddenSize <| List.replicate inputSize 0.0
                     BiasPrevDeltas = List.replicate hiddenSize 0.0
                     NetOutputs = List.empty
 }
 SecondHiddenLayer = {
                      Inputs = List.empty
                      Weights = ((*) hiddenSize hiddenSize) |> listRandElems |> List.chunkBySize hiddenSize
                      Bias = listRandElems hiddenSize
                      Gradients = List.empty
                      PrevDeltas = List.replicate hiddenSize <| List.replicate hiddenSize 0.0
                      BiasPrevDeltas = List.replicate hiddenSize 0.0
                      NetOutputs = List.empty
 }
 OutputLayer = {
                Inputs = List.empty
                Weights = ((*) hiddenSize outputSize) |> listRandElems |> List.chunkBySize hiddenSize
                Bias = listRandElems outputSize
                Gradients = List.empty
                PrevDeltas = List.replicate outputSize <| List.replicate hiddenSize 0.0
                BiasPrevDeltas = List.replicate outputSize 0.0
                NetOutputs = List.empty
 }
 TargetOutputs = List.replicate outputSize 0.0
}

let alldata =
 [
  (*inputs*)   (* expected outputs. *)
  [ 0.1; 0.2;  0.3; 0.4; 0.5; 0.6 ]
  [ 0.2; 0.3;  0.4; 0.5; 0.6; 0.7 ]
 ]

let kfold = 1
let epoch = 500

printfn "Training..."
let trained = train epoch kfold network alldata