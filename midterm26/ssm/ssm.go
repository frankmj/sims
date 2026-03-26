// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// State Space Model (SSM) for sequence prediction.
// CLPS1492 Computational Cognitive Neuroscience -- Midterm
// Please see the associated README.md file for a description of this project.
//
// This file implements a State Space Model (SSM) equivalent to the two-context-
// layer SRN (srn_2ctx).  The key difference is that the two state layers are
// COUPLED: the update of each state dimension depends on both its own previous
// activity AND the previous activity of the other state dimension.  This gives
// the model a full 2x2 state-transition (A) matrix:
//
//	State1(t) = FmHid  * Hidden(t-1) + FmPrv   * State1(t-1) + FmCross  * State2(t-1)
//	State2(t) = FmHid2 * Hidden(t-1) + FmPrv2  * State2(t-1) + FmCross2 * State1(t-1)
//
// In the SRN / srn_2ctx the off-diagonal terms (FmCross, FmCross2) are fixed at
// zero, meaning the two context layers evolve independently.  Setting non-zero
// cross-coupling values here turns the model into a proper linear SSM.

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

//go:embed zeroth.tsv first.tsv third.tsv
var content embed.FS

// PatsType is the type of training patterns
type PatsType int32 //enums:enum

const (
	// zeroth order sequence
	Zeroth PatsType = iota

	// first order sequence
	First

	// 3rd order sequence
	Third
)

func main() {
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
				"Path.WtScale.Rel":       "0.3",
			}},
		{Sel: "Layer", Desc: "needs some special inhibition and learning params",
			Params: params.Params{
				"Layer.Learn.AvgL.Gain":    "1.5", // this is critical! 2.5 def doesn't work
				"Layer.Inhib.Layer.Gi":     "1.3",
				"Layer.Inhib.ActAvg.Init":  "0.5",
				"Layer.Inhib.ActAvg.Fixed": "true",
				"Layer.Act.Gbar.L":         "0.1",
			}},
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
	NRuns int `default:"10" min:"1"`

	// total number of epochs per run
	NEpochs int `default:"400"`

	// stop run after this number of perfect, zero-error epochs.
	NZero int `default:"5"`

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
	// 3rd order training patterns
	Third *table.Table `new-window:"+" display:"no-inline"`

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

	// FmHid is the proportion of State1's new activity taken from the hidden layer (B1 in SSM)
	FmHid float32 `desc:"proportion of State1 activity taken from the hidden layer (B1)"`

	// FmPrv is the proportion of State1's previous activity retained (A11 diagonal of SSM A matrix)
	FmPrv float32 `desc:"proportion of State1 previous activity retained (A11)"`

	// FmHid2 is the proportion of State2's new activity taken from the hidden layer (B2 in SSM)
	FmHid2 float32 `desc:"proportion of State2 activity taken from the hidden layer (B2)"`

	// FmPrv2 is the proportion of State2's previous activity retained (A22 diagonal of SSM A matrix)
	FmPrv2 float32 `desc:"proportion of State2 previous activity retained (A22)"`

	// FmCross is the cross-coupling from State2 into State1's update (A12 off-diagonal of SSM A matrix)
	FmCross float32 `desc:"cross-coupling weight from State2 into State1 update (A12)"`

	// FmCross2 is the cross-coupling from State1 into State2's update (A21 off-diagonal of SSM A matrix)
	FmCross2 float32 `desc:"cross-coupling weight from State1 into State2 update (A21)"`

	TmpVals1 []float32 `display:"-"`
	TmpVals2 []float32 `display:"-"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.FmHid = 1
	ss.FmPrv = 0
	ss.FmHid2 = 1
	ss.FmPrv2 = 0
	ss.FmCross = 0
	ss.FmCross2 = 0

	econfig.Config(&ss.Config, "config.toml")
	ss.Patterns = Zeroth
	ss.Net = leabra.NewNetwork("SSMNet")

	ss.Params.Config(ParamSets, "", "", ss.Net)
	ss.Stats.Init()
	ss.Zeroth = &table.Table{}
	ss.First = &table.Table{}
	ss.Third = &table.Table{}
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.Context.Defaults()
}

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

func (ss *Sim) OpenPatterns() {
	ss.Zeroth.SetMetaData("name", "Zeroth")
	ss.Zeroth.SetMetaData("desc", "zeroth order training patterns")
	errors.Log(ss.Zeroth.OpenFS(content, "zeroth.tsv", table.Tab))

	ss.First.SetMetaData("name", "First")
	ss.First.SetMetaData("desc", "first order training patterns")
	errors.Log(ss.First.OpenFS(content, "first.tsv", table.Tab))

	ss.Third.SetMetaData("name", "Third")
	ss.Third.SetMetaData("desc", "third order training patterns")
	errors.Log(ss.Third.OpenFS(content, "third.tsv", table.Tab))
}

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

	inp := net.AddLayer2D("Input", 1, 6, leabra.InputLayer)
	hid := net.AddLayer2D("Hidden", 6, 5, leabra.SuperLayer)
	out := net.AddLayer2D("Output", 1, 6, leabra.TargetLayer)

	// State1 is the first SSM state dimension.  It receives from the hidden layer
	// (B1 matrix) and from its own previous activity (A11) as well as from State2
	// (A12 cross-coupling).  All updates are computed manually in ApplyInputs.
	state1 := net.AddLayer2D("State1", 6, 5, leabra.InputLayer)

	// State2 is the second SSM state dimension with independent B2/A22 parameters
	// and cross-coupling A21 from State1.
	state2 := net.AddLayer2D("State2", 6, 5, leabra.InputLayer)

	full := paths.NewFull()

	net.ConnectLayers(inp, hid, full, leabra.ForwardPath)
	net.BidirConnectLayers(hid, out, full)
	net.ConnectLayers(state1, hid, full, leabra.ForwardPath)
	net.ConnectLayers(state2, hid, full, leabra.ForwardPath)

	out.PlaceAbove(hid)
	state1.PlaceRightOf(inp, 2)
	state2.PlaceRightOf(state1, 2)

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

// ApplyInputs applies input patterns from the environment and updates the SSM
// state layers.  The state update implements the full SSM equations:
//
//	State1(t) = FmHid  * Hidden(t-1) + FmPrv   * State1(t-1) + FmCross  * State2(t-1)
//	State2(t) = FmHid2 * Hidden(t-1) + FmPrv2  * State2(t-1) + FmCross2 * State1(t-1)
func (ss *Sim) ApplyInputs() {

	ctx := &ss.Context
	net := ss.Net
	// net.InitActs() is commented out so that activations carry over between trials,
	// which is required for the state layer mechanism to work correctly.
	// net.InitActs()

	ev := ss.Envs.ByMode(ctx.Mode).(*env.FixedTable)
	ev.Step()

	net.InitExt()

	lays := net.LayersByType(leabra.InputLayer, leabra.TargetLayer)

	ss.Stats.SetString("TrialName", ev.TrialName.Cur)
	for _, lnm := range lays {
		// Skip the SSM state layers: they are InputLayer type but are not part of the
		// training table and are updated manually from the hidden layer below.
		if lnm == "State1" || lnm == "State2" {
			continue
		}
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

	// Collect hidden layer's previous plus-phase activity (Hidden(t-1)).
	hid := net.LayerByName("Hidden")
	hid.UnitValues(&ss.TmpVals1, "ActP", 0)

	// Collect State1 and State2 previous plus-phase activity for cross-coupling.
	s1Lay := net.LayerByName("State1")
	s2Lay := net.LayerByName("State2")
	s1Lay.UnitValues(&ss.TmpVals2, "ActP", 0) // State1(t-1)

	// Temporary storage for State2(t-1) before we overwrite State1.
	s2Prev := make([]float32, len(s2Lay.Neurons))
	s2Lay.UnitValues(&s2Prev, "ActP", 0) // State2(t-1)

	// State1 SSM update: A11*State1(t-1) + A12*State2(t-1) + B1*Hidden(t-1)
	clr1, set1, toTarg1 := s1Lay.ApplyExtFlags()
	for i := range s1Lay.Neurons {
		ext := ss.FmHid*ss.TmpVals1[i] +
			ss.FmPrv*ss.TmpVals2[i] +
			ss.FmCross*s2Prev[i]
		s1Lay.ApplyExtValue(i, ext, clr1, set1, toTarg1)
	}

	// State2 SSM update: A22*State2(t-1) + A21*State1(t-1) + B2*Hidden(t-1)
	clr2, set2, toTarg2 := s2Lay.ApplyExtFlags()
	for i := range s2Lay.Neurons {
		ext := ss.FmHid2*ss.TmpVals1[i] +
			ss.FmPrv2*s2Prev[i] +
			ss.FmCross2*ss.TmpVals2[i]
		s2Lay.ApplyExtValue(i, ext, clr2, set2, toTarg2)
	}
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
	case Third:
		trn.Table = table.NewIndexView(ss.Third)
		tst.Table = table.NewIndexView(ss.Third)
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
	title := "State Space Model (SSM)"
	ss.GUI.MakeBody(ss, "SSM", title, `SSM midterm -- state space model with two coupled state dimensions`)
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
