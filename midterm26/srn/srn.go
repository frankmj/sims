// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Simple Recurrent Network (SRN)
// CLPS1492 Computational Cognitive Neuroscience -- Midterm
// Please see the associated README.md file for a description of this project.

package main

//go:generate core generate -add-types

import (
	"embed"
	// Uncomment the below line to import "fmt" that supports printing in Go. This will prove useful for debugging. 
	//"fmt"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/lab/base/randx" 
	"cogentcore.org/core/core"
	"cogentcore.org/core/enums" 
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"github.com/emer/etensor/tensor/table"
	"cogentcore.org/core/tree"
	"github.com/emer/emergent/v2/econfig"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/elog"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/estats"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/netview"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/paths"
	"github.com/emer/leabra/v2/leabra"
)

//go:embed zeroth.tsv first.tsv custom.tsv
var content embed.FS

// PatsType is the type of training patterns
type PatsType int32 //enums:enum

const (
	// zeroth order sequence
	Zeroth PatsType = iota

	// first order sequence
	First

	// custom nth order sequence
	Custom
)

// LearnType is the type of learning to use
type LearnType int32 //enums:enum


// ****************************************************************************
// Landmark 1
//
// The main() function below is where the go program for our project starts.
// A good way to understand how a program works (in general, not just re. Emergent)
// is to start where it starts and step through its commands. It's a good idea
// to take a little time in this project to look over things, even when doing so
// isn't strictly necessary for completing the assignment, because the extra
// familiarity will make getting going on your final project far easier.
//
// The basic syntax being used is object.method(). This is standard
// object-oriented programming, wherein we keep data ("fields") and functions
// that run using that data ("methods") packaged together in "objects". (Objects
// can also have other objects inside them.) We also try to name things informatively.
// "sim" is an object that controls our whole emergent simulation, for example.
// sim.New() tells the object sim to create blank versions of the variables a
// sim type object should have.
//
// Take a minute to check out the code below. It's not critical that you understand
// every piece of it, just do your best to guess what things might mean.
//
// Once you've thought about this for a minute, you might reasonably suspect that
// sim.New() is going to do a lot of book keeping and variable initialization,
// whereas sim.Config() might start setting some of those variables. We'll check
// these out next.
//
// To go there, you should use your text editor to find New(). Alternatively, you can
// search "Landmark 2", but going through the process of searching for the functions
// is more similar to the process you would use in figuring things out yourself.
//
// ****************************************************************************


func main() {
	
	// sim is defined here, which is convenient.
	// sim is the overall state for this simulation
	sim := &Sim{}
	
	sim.New()
	sim.ConfigAll()
	sim.RunGUI()
}

// ParamSets is the default set of parameters.
// Base is always applied, and others can be optionally
// selected to apply on top of that.
var ParamSets = params.Sets{
	"Base": {
		{Sel: "Path", Desc: "no extra learning factors",
			Params: params.Params{
				"Path.Learn.Norm.On":     "true",
				"Path.Learn.Momentum.On": "true",
				"Path.Learn.WtBal.On":    "false",
				"Path.WtScale.Rel": "0.3",
			}},
		{Sel: "Layer", Desc: "needs some special inhibition and learning params",
			Params: params.Params{
				"Layer.Learn.AvgL.Gain":    "1.5", // this is critical! 2.5 def doesn't work
				"Layer.Inhib.Layer.Gi":     "1.3",
				"Layer.Inhib.ActAvg.Init":  "0.5",
				"Layer.Inhib.ActAvg.Fixed": "true",
				"Layer.Act.Gbar.L":         "0.1",
			}},
		// {Sel: ".BackPath", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
		// 	Params: params.Params{
		// 		"Path.WtScale.Rel": "0.3",
		// 	}},
		{Sel: "#Output", Desc: "output definitely needs lower inhib -- true for smaller layers in general",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "1.4",
			}},
	},
	"Hebbian": {
		{Sel: "Path", Desc: "",
			Params: params.Params{
				"Path.Learn.XCal.MLrn":    "0",
				"Path.Learn.XCal.SetLLrn": "true",
				"Path.Learn.XCal.LLrn":    "1",
			}},
	},
	"ErrorDriven": {
		{Sel: "Path", Desc: "",
			Params: params.Params{
				"Path.Learn.XCal.MLrn":    "1",
				"Path.Learn.XCal.SetLLrn": "true",
				"Path.Learn.XCal.LLrn":    "0",
			}},
	},
}

// Config has config parameters related to running the sim
type Config struct {
	// total number of runs to do when running Train
	NRuns int `default:"1" min:"1"`

	// total number of epochs per run
	NEpochs int `default:"400"`

	// stop run after this number of perfect, zero-error epochs.
	NZero int `default:"20"`

	// how often to run through all the test patterns, in terms of training epochs.
	// can use 0 or -1 for no testing.
	TestInterval int `default:"5"`
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {

	// select which type of patterns to use
	Patterns PatsType

	// Config contains misc configuration parameters for running the sim
	Config Config `new-window:"+" display:"no-inline"`

	// the network -- click to view / edit parameters for layers, paths, etc
	Net *leabra.Network `new-window:"+" display:"no-inline"`

	// network parameter management
	Params emer.NetParams `display:"-"`

	// zeroth order training patterns
	Zeroth *table.Table `new-window:"+" display:"no-inline"`
	// first order training patterns
	First *table.Table `new-window:"+" display:"no-inline"`
	// custom n-th order training patterns
	Custom *table.Table `new-window:"+" display:"no-inline"`

	// contains looper control loops for running sim
	Loops *looper.Stacks `new-window:"+" display:"no-inline"`

	// contains computed statistic values
	Stats estats.Stats `new-window:"+"`

	// Contains all the logs and information about the logs.'
	Logs elog.Logs `new-window:"+"`

	// Environments
	Envs env.Envs `new-window:"+" display:"no-inline"`

	// leabra timing parameters and state
	Context leabra.Context `new-window:"+"`

	// netview update parameters
	ViewUpdate netview.ViewUpdate `display:"add-fields"`

	// manages all the gui elements
	GUI egui.GUI `display:"-"`

	// a list of random seeds to use for each run
	RandSeeds randx.Seeds `display:"-"`

	// Uncomment the below lines to add FmHid and FmPrv to the Sim struct.
	// FmHid        float32           `desc:"percentage of the context layer activity that will be taken directly from the hidden layer"`
	// FmPrv        float32           `desc:"percentage of the context layer activity that will be kept"`

	TmpVals1      []float32 `display:"-"`
	TmpVals2      []float32 `display:"-"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {

	// -----------------------------------
	// Note: When you uncomment these, they become available in
	// the control panel. For future reference, this is true in
	// general of items such as parameters which you might like
	// to put in the control panel.
	//
	// For those parameters which you don't add or modify here,
	// the ParamSet field in the control panel can also be used
	// to modify parameters. These will be reset when Init is
	// called from within the GUI however!
	//
	// ss.FmHid = 1
	// ss.FmPrv = 0
	// -----------------------------------


	econfig.Config(&ss.Config, "config.toml")
	ss.Patterns = Zeroth
	ss.Net = leabra.NewNetwork("HiddenNet")
	
	ss.Params.Config(ParamSets, "", "", ss.Net)
	ss.Stats.Init()
	ss.Zeroth = &table.Table{}
	ss.First = &table.Table{}
	ss.Custom = &table.Table{}
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.Context.Defaults()
}

// ****************************************************************************
// Landmark 2
//
// It looks like New() was the place to be! 
// Take a minute to look through the declarations above starting with
// "type Sim struct {...}". Thats where all the program variables are kept.
//
// It also looks like New() is doing more or less what we thought: Creating what
// look like empty variables. As for the ConfigAll() function, it takes an argument
// called 'ss', which has to be a Sim object, and then it just calls a bunch more
// specific configuration functions. In this sense, it's a "wrapper": a function
// that allows you to conveniently call a bunch of things, but doesn't have a lot
// of substantial manipulation logic happening within it's own highest "scope"
// (i.e. the stuff between the {} in ConfigAll(){...}.
//
// Try reading through the functions ConfigEnv() and ConfigNet() and guessing what
// the various statements might mean. Env is often shorthand for "Environment",
// Net is probably "Network", Logs might be "Logging", etc.
//
// Having done that, you may have noticed that the whole section below on
// configurations is demarcated with a series of '/'s. It looks like whoever
// made this file used that convention to partition off big sections. You
// should search for those sections using your text editor's 'find' function,
// and a long string of backslashes, then look around in them a bit.
//
// When you're done, you should return to the instructions in the README.
// ****************************************************************************

//////////////////////////////////////////////////////////////////////////////
// 		Configs

// ConfigAll configures all the elements using the standard functions
func (ss *Sim) ConfigAll() {
	ss.OpenPatterns()
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
	ss.ConfigLoops()
}


// ****************************************************************************
// Section 1b: Input patterns
//
// The function below reads a data file which determines the task the network
// should perform. What this means is that the file should specify what the
// input to the network is and, if applicable, what the correct output would be.
//
// Open the file named "zeroth.tsv" in the same directory as the
// srn.go program. You should see the following lines:
//
// _H:	$Name	%Input[2:0,0]<2:1,6>	%Input[2:0,1]	%Input[2:0,2]	%Input[2:0,3]	%Input[2:0,4]	%Input[2:0,5]	%Output[2:0,0]<2:1,6>	%Output[2:0,1]	%Output[2:0,2]	%Output[2:0,3]	%Output[2:0,4]	%Output[2:0,5]
// _D:	A	1	0	0	0	0	0	0	1	0	0	0	0
// _D:	B	0	1	0	0	0	0	0	0	1	0	0	0
// _D:	C	0	0	1	0	0	0	0	0	0	1	0	0
//
//
// The first line is what's called metadata, a "header" which specifies the format
// of the subsequent lines. What these subsequent lines (starting with _D:) specify
// are the data, organized in columns given by the header.
//
// The first three lines of a sequence task are given for you. You should add three
// further lines specifying the input and output for three additional stimuli named
// D, E, and F. They should follow the same pattern above, where each input predicts
// the next input as output. In the A column, for example, the 1st of 6 inputs is a
// '1' and the 2nd of 6 outputs is a '1', whereas the rest are zero. This says the
// 'A' input should make the network give the 'B' output.
//
// TWO VERY IMPORTANT NOTES:
// 1) When you copy and modify the first few lines in the file, and save it, MAKE
// SURE your text editor is NOT converting TABS to SPACES! The file will not be read
// properly with spaces.
//
// 2) Once you've edited the zeroth.tsv file, you can verify that the simulation program
// is reading the file correctly by building the simulation. On the simulation window, click on 
// Zeroth training data (this should appear on the left-hand sidebar. Click on "Zeroth" that 
// appears beside the pencil icon). You should now be able to see the input-output for every
// trial in zeroth.tsv. If you see a black window, there program is not able to read the 
// data file. Try debugging the data file.  
// 
// 
// After editing zeroth.tsv, you should be able to train the network on zeroth order sequence.
// Ensure that "Zeroth" is selected as "Patterns" on the left-hand sidebar to train the network
// on this sequence. At a later stage, you will be asked to train the network on first, second and 
// other higher order sequences. You are already given a sample first order sequence in first.tsv. 
// You can select this from the GUI to train the network on a first order sequence. 
// In order to train the SRN with higher-order sequences you can use custom.tsv file. 
// You can edit custom.tsv to train the network on your custom n-th order sequence. 
// Don't forget to select "Custom" from the GUI when training the network on this sequence.  
//

func (ss *Sim) OpenPatterns() {

	// read and load zeroth.tsv file
	ss.Zeroth.SetMetaData("name", "Zeroth")
	ss.Zeroth.SetMetaData("desc", "zeroth order training patterns")
	errors.Log(ss.Zeroth.OpenFS(content, "zeroth.tsv", table.Tab))

	// read and load first.tsv file
	ss.First.SetMetaData("name", "First")
	ss.First.SetMetaData("desc", "first order training patterns")
	errors.Log(ss.First.OpenFS(content, "first.tsv", table.Tab))

	// read and load custom.tsv file
	ss.Custom.SetMetaData("name", "Custom")
	ss.Custom.SetMetaData("desc", "custom n-th order training patterns")
	errors.Log(ss.Custom.OpenFS(content, "custom.tsv", table.Tab))
}

//
// This is the end of section 1b. Now build the program using `core run`, 
// run the file, and continue with the README.
// ****************************************************************************

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	var trn, tst *env.FixedTable
	if len(ss.Envs) == 0 {
		trn = &env.FixedTable{}
		tst = &env.FixedTable{}
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*env.FixedTable)
		tst = ss.Envs.ByMode(etime.Test).(*env.FixedTable)
	}

	// note: names must be standard here!
	trn.Name = etime.Train.String()
	trn.Config(table.NewIndexView(ss.Zeroth))
	trn.Sequential = true
	trn.Validate()

	tst.Name = etime.Test.String()
	tst.Config(table.NewIndexView(ss.Zeroth))
	tst.Sequential = true
	tst.Validate()

	trn.Init(0)
	tst.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(trn, tst)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	// ****************************************************************************
	// Section 1a: Setting up a three layer network:
	//
	// Here you will need to 'write' some code to genenerate layers in the network,
	// then connect them. The basic instruction for adding a layer is:
	//
	// 	layer := net.AddLayer2D(name, num_rows, num_columns, type)
	//
	// where the items in () are variables defining the layer. A name is a string
	// variable one chooses, num_rows and num_columns are numbers, and type determines
	// whether a layer will have inputs or outputs clamped on in training.
	//
	// STEP 1: Add a hidden layer here. It should be called "Hidden" and the variable
	//         should be 'hid', mimicking 'inp' and 'out'. It should have 6 rows and
	//         5 columns. You can create it by copying 'inp' then modifying args.
	//         Its type should be leabra.SuperLayer.
	//
	// Please insert the additional code directly below inp.

	inp := net.AddLayer2D("Input", 1, 6, leabra.InputLayer)
	out := net.AddLayer2D("Output", 1, 6, leabra.TargetLayer)
	//
	// ****************************************************************************


	full := paths.NewFull()

	// ****************************************************************************
	// STEP 2:
	// Connect the input to the hidden layer (which you just created) with feed-forward projections.
	// Connect the hidden layer to the output layer with bidirectional connections.
	//
	// The syntax to do this is:
	//
	// 	net.ConnectLayers(from, to, full, leabra.ForwardPath)
	// 	net.BidirConnectLayers(from, to, full)
	//
	// Where "from" and "to" are the variables you created in step 2. The inputs full 
	// (of type paths.NewFull() defined just above this description of STEP 2)
	// and emer.Forward tell the ConnectLayers() function that the connectivity should be
	// all-to-all (each unit sends to every other) and feed-forward (units in the
	// 'to' layer don't send synapses back).
	//
	// Please remove the following line, connecting inp to out, and replace it with
	// the connections we've just described.
	//
	net.ConnectLayers(inp, out, full, leabra.ForwardPath)
	//
	// ****************************************************************************


	// ****************************************************************************
	// STEP 3:
	// Make the GUI layout of the network neater. 
	// Once the context layer is introduced, the GUI might render multiple layers
	// on top of each other. While this does not affect the functionality of the SRN, 
	// it is not so neat to look at. Here we will specify the relative location of the
	// layers to improve the GUI layout of the network.
	//
	// layer1.PlaceAbove(layer2) is used to place layer1 above layer2.
	// 
	// Uncomment the following line to adjust the GUI layout for the output layer. 
	//
	// out.PlaceAbove(hid) // TODO: remove
	//
	// Once you've done this. Please proceed to Section 1b (by searching this file.)
	// ****************************************************************************


	// ****************************************************************************
	// Section 3: Context Layer GUI Placement
	// You will need to visit this part of the code in Section 3 (from the README). 
	// Ignore this part of the code until you reach Section 3.
	//
	// Just like what we did earlier, we will make the GUI layout of the network neater again. 
	// layer1.PlaceRightOf(layer2, 2) is used to place layer1 on the right side of layer2.
	// The integer denotes the separation. 
	// 
	// Uncomment the following lines to adjust the GUI layout for the context layer. 
	//
	// context.PlaceRightOf(inp, 2) // TODO: remove
	//
	// Once you've done this. Please proceed to the README file.
	// ****************************************************************************

	net.Build()
	net.Defaults()
	ss.ApplyParams()
	net.InitWeights()
}

func (ss *Sim) ApplyParams() {
	ss.Params.SetAll()
	ss.Params.SetAllSheet("ErrorDriven")
	if ss.Loops != nil {
		trn := ss.Loops.Stacks[etime.Train]
		trn.Loops[etime.Run].Counter.Max = ss.Config.NRuns
		trn.Loops[etime.Epoch].Counter.Max = ss.Config.NEpochs
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // in case user interactively changes tag
	ss.Loops.ResetCounters()
	ss.InitRandSeed(0)
	ss.ConfigEnv() // re-config env just in case a different set of patterns was
	ss.GUI.StopNow = false
	ss.ApplyParams()
	ss.NewRun()
	ss.ViewUpdate.RecordSyns()
	ss.ViewUpdate.Update()
}

// InitRandSeed initializes the random seed based on current training run number
func (ss *Sim) InitRandSeed(run int) {
	ss.RandSeeds.Set(run)
	ss.RandSeeds.Set(run, &ss.Net.Rand)
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	ls := looper.NewStacks()

	trls := 10


	ls.AddStack(etime.Train).
		AddTime(etime.Run, ss.Config.NRuns).
		AddTime(etime.Epoch, ss.Config.NEpochs).
		AddTime(etime.Trial, trls).
		AddTime(etime.Cycle, 100)

	ls.AddStack(etime.Test).
		AddTime(etime.Epoch, 1).
		AddTime(etime.Trial, trls).
		AddTime(etime.Cycle, 100)

	leabra.LooperStdPhases(ls, &ss.Context, ss.Net, 75, 99)                // plus phase timing
	leabra.LooperSimCycleAndLearn(ls, ss.Net, &ss.Context, &ss.ViewUpdate) // std algo code

	for m, _ := range ls.Stacks {
		mode := m // For closures
		stack := ls.Stacks[mode]
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			ss.ApplyInputs()
		})
	}

	ls.Loop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	// Train stop early condition
	ls.Loop(etime.Train, etime.Epoch).IsDone.AddBool("NZeroStop", func() bool {
		// This is calculated in TrialStats
		stopNz := ss.Config.NZero
		if stopNz <= 0 {
			stopNz = 2
		}
		curNZero := ss.Stats.Int("NZero")
		stop := curNZero >= stopNz
		return stop
	})

	// Add Testing
	trainEpoch := ls.Loop(etime.Train, etime.Epoch)
	trainEpoch.OnStart.Add("TestAtInterval", func() {
		if (ss.Config.TestInterval > 0) && ((trainEpoch.Counter.Cur+1)%ss.Config.TestInterval == 0) {
			// Note the +1 so that it doesn't occur at the 0th timestep.
			ss.TestAll()
		}
	})

	/////////////////////////////////////////////
	// Logging

	ls.Loop(etime.Test, etime.Epoch).OnEnd.Add("LogTestErrors", func() {
		leabra.LogTestErrors(&ss.Logs)
	})
	ls.AddOnEndToAll("Log", func(mode enums.Enum, time enums.Enum) {  
    ss.Log(mode.(etime.Modes), time.(etime.Times))
	})
	leabra.LooperResetLogBelow(ls, &ss.Logs)
	ls.Loop(etime.Train, etime.Run).OnEnd.Add("RunStats", func() {
		ss.Logs.RunStats("PctCor", "FirstZero", "LastZero")
	})

	////////////////////////////////////////////
	// GUI

	leabra.LooperUpdateNetView(ls, &ss.ViewUpdate, ss.Net, ss.NetViewCounters)
	leabra.LooperUpdatePlots(ls, &ss.GUI)

	ss.Loops = ls
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {

	ctx := &ss.Context
	net := ss.Net
	// Commment out line below (ie insert // in front) when you get to Section 3 and start using the SRN
        net.InitActs() // reinitialize all activations/state every trial

	ev := ss.Envs.ByMode(ctx.Mode).(*env.FixedTable)
	ev.Step()

	net.InitExt()

	lays := net.LayersByType(leabra.InputLayer, leabra.TargetLayer)
	
	ss.Stats.SetString("TrialName", ev.TrialName.Cur)
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm)
		pats := ev.State(ly.Name)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}

	out := ss.Net.LayerByName("Output")
	if ctx.Mode == etime.Test {
		out.Type = leabra.CompareLayer // don't clamp plus phase
	} else {
		out.Type = leabra.TargetLayer
	}

	// **********************************
	// Insert context-layer code snippet from the README here:

	// **********************************
	
}

func (ss *Sim) UpdateEnv() {
	trn := ss.Envs.ByMode(etime.Train).(*env.FixedTable)
	tst := ss.Envs.ByMode(etime.Test).(*env.FixedTable)
	switch ss.Patterns {
	case Zeroth:
		trn.Table = table.NewIndexView(ss.Zeroth)
		tst.Table = table.NewIndexView(ss.Zeroth)
	case First:
		trn.Table = table.NewIndexView(ss.First)
		tst.Table = table.NewIndexView(ss.First)
	case Custom:
		trn.Table = table.NewIndexView(ss.Custom)
		tst.Table = table.NewIndexView(ss.Custom)
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ctx := &ss.Context
	ss.InitRandSeed(ss.Loops.Loop(etime.Train, etime.Run).Counter.Cur)
	ss.UpdateEnv()
	ss.Envs.ByMode(etime.Train).Init(0)
	ss.Envs.ByMode(etime.Test).Init(0)
	ctx.Reset()
	ctx.Mode = etime.Train
	ss.Net.InitWeights()
	ss.InitStats()
	ss.StatCounters()
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
	ss.ApplyParams()
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.Envs.ByMode(etime.Test).Init(0)
	ss.Loops.ResetAndRun(etime.Test)
	ss.Loops.Mode = etime.Train // Important to reset Mode back to Train because this is called from within the Train Run.
}

/////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetFloat("SSE", 0.0)
	ss.Stats.SetString("TrialName", "")
	ss.Logs.InitErrStats() // inits TrlErr, FirstZero, LastZero, NZero
}

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them for ViewUpdate.Text
func (ss *Sim) StatCounters() {
	ctx := &ss.Context
	mode := ctx.Mode
	ss.Loops.Stacks[mode].CountersToStats(&ss.Stats)
	// always use training epoch..
	trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	ss.Stats.SetInt("Epoch", trnEpc)
	trl := ss.Stats.Int("Trial")
	ss.Stats.SetInt("Trial", trl)
	ss.Stats.SetInt("Cycle", int(ctx.Cycle))
}

func (ss *Sim) NetViewCounters(tm etime.Times) {
	if ss.ViewUpdate.View == nil {
		return
	}
	if tm == etime.Trial {
		ss.TrialStats() // get trial stats for current di
	}
	ss.StatCounters()
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "TrialName", "Cycle", "SSE", "TrlErr"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
	out := ss.Net.LayerByName("Output")

	sse, avgsse := out.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5
	ss.Stats.SetFloat("SSE", sse)
	ss.Stats.SetFloat("AvgSSE", avgsse)
	if sse > 0 {
		ss.Stats.SetFloat("TrlErr", 1)
	} else {
		ss.Stats.SetFloat("TrlErr", 0)
	}
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")

	ss.Logs.AddStatAggItem("SSE", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("AvgSSE", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddErrStatAggItems("TrlErr", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddCopyFromFloatItems(etime.Train, []etime.Times{etime.Epoch, etime.Run}, etime.Test, etime.Epoch, "Tst", "SSE", "AvgSSE")

	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddLayerTensorItems(ss.Net, "ActM", etime.Test, etime.Trial, "InputLayer", "SuperLayer", "TargetLayer")
	ss.Logs.AddLayerTensorItems(ss.Net, "Targ", etime.Test, etime.Trial, "TargetLayer")

	ss.Logs.PlotItems("SSE", "FirstZero", "LastZero")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Trial)
	ss.Logs.NoPlot(etime.Test, etime.Run)
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	ctx := &ss.Context
	if mode != etime.Analyze {
		ctx.Mode = mode // Also set specifically in a Loop callback.
	}
	dt := ss.Logs.Table(mode, time)
	if dt == nil {
		return
	}
	row := dt.Rows

	switch {
	case time == etime.Cycle:
		return
	case time == etime.Trial:
		ss.TrialStats()
		ss.StatCounters()
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc

	if mode == etime.Test {
		ss.GUI.UpdateTableView(etime.Test, etime.Trial)
	}
}

//////////////////////////////////////////////////////////////////////
// 		GUI

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "Simple Recurrent Network"
	ss.GUI.MakeBody(ss, "SRN", title, `SRN midterm`)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("Network")
	nv.Options.MaxRecs = 300
	nv.SetNet(ss.Net)
	nv.Options.PathWidth = 0.005
	ss.ViewUpdate.Config(nv, etime.GammaCycle, etime.GammaCycle)
	ss.GUI.ViewUpdate = &ss.ViewUpdate
	nv.Current()

	nv.SceneXYZ().Camera.Pose.Pos.Set(0.1, 1.5, 4) // more "head on" than default which is more "top down"
	nv.SceneXYZ().Camera.LookAt(math32.Vec3(0.1, 0.1, 0), math32.Vec3(0, 1, 0))

	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.AddTableView(&ss.Logs, etime.Test, etime.Trial)

	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	// ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Init", Icon: icons.Update,
	// 	Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
	// 	Active:  egui.ActiveStopped,
	// 	Func: func() {
	// 		ss.Init()
	// 		ss.GUI.UpdateWindow()
	// 	},
	// })

	ss.GUI.AddLooperCtrl(p, ss.Loops)

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Test Init", Icon: icons.Update,
		Tooltip: "Initialize testing to start over -- if Test Step doesn't work, then do this.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Loops.ResetCountersByMode(etime.Test)
		},
	})

	////////////////////////////////////////////////
	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Reset RunLog",
		Icon:    icons.Reset,
		Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.Logs.ResetLog(etime.Train, etime.Run)
			ss.GUI.UpdatePlot(etime.Train, etime.Run)
		},
	})
	////////////////////////////////////////////////
	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "New Seed",
		Icon:    icons.Add,
		Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.RandSeeds.NewSeeds()
		},
	})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "README", 
		Icon:    icons.FileMarkdown,
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			core.TheApp.OpenURL("https://canvas.brown.edu/")
		},
	})
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
}
