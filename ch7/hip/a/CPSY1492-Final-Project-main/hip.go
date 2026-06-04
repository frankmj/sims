// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// hip runs a hippocampus model on the AB-AC paired associate learning task.
package main

//go:generate core generate -add-types

import (
	"embed"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"strings"
	"os"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/core"
	"cogentcore.org/core/enums"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/tree"
	"cogentcore.org/lab/base/randx"
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
	"github.com/emer/emergent/v2/patgen"
	"github.com/emer/emergent/v2/paths"
	"github.com/emer/etensor/plot/plotcore"
	"github.com/emer/etensor/tensor/stats/split"
	"github.com/emer/etensor/tensor/table"
	"github.com/emer/leabra/v2/leabra"
)

//go:embed train_ab.tsv train_ac.tsv test_ab.tsv test_ac.tsv test_lure.tsv
var content embed.FS

//go:embed *.png README.md
var readme embed.FS

func main() {
	sim := &Sim{}
	sim.New()
	sim.ConfigAll()
	sim.RunGUI()
}

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	"Base": {
		{Sel: "Path", Desc: "keeping default params for generic prjns",
			Params: params.Params{
				"Path.Learn.Momentum.On": "true",
				"Path.Learn.Norm.On":     "true",
				"Path.Learn.WtBal.On":    "false",
			}},
		{Sel: ".EcCa1Path", Desc: "encoder projections -- no norm, moment",
			Params: params.Params{
				"Path.Learn.Lrate":        "0.04",
				"Path.Learn.Momentum.On":  "false",
				"Path.Learn.Norm.On":      "false",
				"Path.Learn.WtBal.On":     "true",
				"Path.Learn.XCal.SetLLrn": "false", // using bcm now, better
			}},
		{Sel: ".HippoCHL", Desc: "hippo CHL projections -- no norm, moment, but YES wtbal = sig better",
			Params: params.Params{
				"Path.CHL.Hebb":          "0.05",
				"Path.Learn.Lrate":       "0.2",
				"Path.Learn.Momentum.On": "false",
				"Path.Learn.Norm.On":     "false",
				"Path.Learn.WtBal.On":    "true",
			}},
		{Sel: ".PPath", Desc: "perforant path, new Dg error-driven EcCa1Path prjns",
			Params: params.Params{
				"Path.Learn.Momentum.On": "false",
				"Path.Learn.Norm.On":     "false",
				"Path.Learn.WtBal.On":    "true",
				"Path.Learn.Lrate":       "0.15",
			}},
		{Sel: "#CA1ToECout", Desc: "extra strong from CA1 to ECout",
			Params: params.Params{
				"Path.WtScale.Abs": "4.0",
			}},
		{Sel: "#InputToECin", Desc: "one-to-one input to EC",
			Params: params.Params{
				"Path.Learn.Learn": "false",
				"Path.WtInit.Mean": "0.8",
				"Path.WtInit.Var":  "0.0",
			}},
		{Sel: "#ECoutToECin", Desc: "one-to-one out to in",
			Params: params.Params{
				"Path.Learn.Learn": "false",
				"Path.WtInit.Mean": "0.9",
				"Path.WtInit.Var":  "0.01",
				"Path.WtScale.Rel": "0.5",
			}},
		{Sel: "#DGToCA3", Desc: "Mossy fibers: strong, non-learning",
			Params: params.Params{
				"Path.Learn.Learn": "false",
				"Path.WtInit.Mean": "0.9",
				"Path.WtInit.Var":  "0.01",
				"Path.WtScale.Rel": "4",
			}},
		{Sel: "#CA3ToCA3", Desc: "CA3 recurrent cons",
			Params: params.Params{
				"Path.WtScale.Rel": "0.1",
				"Path.Learn.Lrate": "0.1",
			}},
		{Sel: "#ECinToDG", Desc: "DG learning is surprisingly critical: maxed out fast, hebbian works best",
			Params: params.Params{
				"Path.Learn.Learn":       "true",
				"Path.CHL.Hebb":          ".5",
				"Path.CHL.SAvgCor":       "0.1",
				"Path.CHL.MinusQ1":       "true",
				"Path.Learn.Lrate":       "0.4",
				"Path.Learn.Momentum.On": "false",
				"Path.Learn.Norm.On":     "false",
				"Path.Learn.WtBal.On":    "true",
			}},
		{Sel: "#CA3ToCA1", Desc: "Schaffer collaterals -- slower, less hebb",
			Params: params.Params{
				"Path.CHL.Hebb":          "0.01",
				"Path.CHL.SAvgCor":       "0.4",
				"Path.Learn.Lrate":       "0.1",
				"Path.Learn.Momentum.On": "false",
				"Path.Learn.Norm.On":     "false",
				"Path.Learn.WtBal.On":    "true",
			}},
		{Sel: ".EC", Desc: "EC layers: layer-level inhibition (2D, no pools)",
			Params: params.Params{
				"Layer.Act.Gbar.L":        ".1",
				"Layer.Inhib.ActAvg.Init": "0.2",
				"Layer.Inhib.Layer.On":    "true",
				"Layer.Inhib.Layer.Gi":    "2.0",
				"Layer.Inhib.Pool.On":     "false",
			}},
		{Sel: "#DG", Desc: "very sparse = high inhibition",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Init": "0.01",
				"Layer.Inhib.Layer.Gi":    "3.8",
			}},
		{Sel: "#CA3", Desc: "sparse = high inhibition",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Init": "0.02",
				"Layer.Inhib.Layer.Gi":    "2.8",
			}},
		{Sel: "#CA1", Desc: "CA1 layer-level inhibition (2D, no pools)",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Init": "0.1",
				"Layer.Inhib.Layer.On":    "true",
				"Layer.Inhib.Layer.Gi":    "2.4",
				"Layer.Inhib.Pool.On":     "false",
			}},
	},
}

// Config has config parameters related to running the sim
type Config struct {
	// total number of runs to do when running Train
	NRuns int `default:"10" min:"1"`

	// total number of epochs per run
	NEpochs int `default:"20"`

	// stop run after this number of perfect, zero-error epochs.
	NZero int `default:"1"`

	// how often to run through all the test patterns, in terms of training epochs.
	// can use 0 or -1 for no testing.
	TestInterval int `default:"1"`

	// StopMem is the threshold for stopping learning.
	StopMem float32 `default:"1"`

	// OverlapPct is the percent overlap between stored patterns (0, 20, 40, 50, 60, 80).
	// Set this in config.toml for each condition you run.
	OverlapPct int `default:"0"`

	// PatternFile is the path to the .tsv file for this overlap condition.
	// e.g. "/path/to/patterns_0.tsv"
	PatternFile string `default:"patterns_0.tsv"`

	// TestPatternFile is the path to the held-out test .tsv file for this overlap condition.
	// e.g. "/path/to/test_patterns_0.tsv"
	// The test file should have partial inputs (cue only) with full ECout targets.
	TestPatternFile string `default:"test_patterns_0.tsv"`
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.
type Sim struct {

	// simulation configuration parameters -- set by .toml config file and / or args
	Config Config `new-window:"+"`

	// the network -- click to view / edit parameters for layers, paths, etc
	Net *leabra.Network `new-window:"+" display:"no-inline"`

	// all parameter management
	Params emer.NetParams `display:"add-fields"`

	// contains looper control loops for running sim
	Loops *looper.Stacks `new-window:"+" display:"no-inline"`

	// contains computed statistic values
	Stats estats.Stats `new-window:"+"`

	// Contains all the logs and information about the logs.
	Logs elog.Logs `new-window:"+"`

	// if true, run in pretrain mode
	PretrainMode bool `display:"-"`

	// pool patterns vocabulary
	PoolVocab patgen.Vocab `display:"-"`

	// Training patterns for this overlap condition
	TrainAB *table.Table `new-window:"+" display:"no-inline"`

	// AC training patterns (unused in overlap study, kept for compatibility)
	TrainAC *table.Table `new-window:"+" display:"no-inline"`

	// AB testing patterns (unused in overlap study, kept for compatibility)
	TestAB *table.Table `new-window:"+" display:"no-inline"`

	// AC testing patterns (unused in overlap study, kept for compatibility)
	TestAC *table.Table `new-window:"+" display:"no-inline"`

	// Lure testing patterns (unused in overlap study, kept for compatibility)
	TestLure *table.Table `new-window:"+" display:"no-inline"`

	// TestAll has the held-out test patterns for this overlap condition.
	// These should be partial-cue inputs with full ECout targets.
	TestAll *table.Table `new-window:"+" display:"no-inline"`

	// Lure pretrain patterns
	PreTrainLure *table.Table `new-window:"+" display:"-"`

	// all training patterns -- for pretrain
	TrainAll *table.Table `new-window:"+" display:"-"`

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
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	econfig.Config(&ss.Config, "config.toml")

	ss.Net = leabra.NewNetwork("Hip")
	ss.Params.Config(ParamSets, "", "", ss.Net)
	ss.Stats.Init()
	ss.Stats.SetInt("Expt", 0)

	ss.PoolVocab = patgen.Vocab{}
	ss.TrainAB = &table.Table{}
	ss.TrainAC = &table.Table{}
	ss.TestAB = &table.Table{}
	ss.TestAC = &table.Table{}
	ss.PreTrainLure = &table.Table{}
	ss.TestLure = &table.Table{}
	ss.TrainAll = &table.Table{}
	ss.TestAll = &table.Table{}
	ss.PretrainMode = false

	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.Context.Defaults()
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// ConfigAll configures all the elements using the standard functions
func (ss *Sim) ConfigAll() {
	ss.OpenPatterns()
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
	ss.ConfigLoops()
}

func (ss *Sim) ConfigEnv() {
	var trn, tst *env.FixedTable
	if len(ss.Envs) == 0 {
		trn = &env.FixedTable{}
		tst = &env.FixedTable{}
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*env.FixedTable)
		tst = ss.Envs.ByMode(etime.Test).(*env.FixedTable)
	}

	trn.Name = etime.Train.String()
	trn.Config(table.NewIndexView(ss.TrainAB))
	trn.Validate()

	tst.Name = etime.Test.String()
	tst.Config(table.NewIndexView(ss.TestAll))
	tst.Sequential = true
	tst.Validate()

	trn.Init(0)
	tst.Init(0)

	ss.Envs.Add(trn, tst)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.SetRandSeed(ss.RandSeeds[0])

	// Layers sized to match your patterns: 40 units each, laid out as [5, 8] (2D).
	// Input/ECin/ECout: 5 rows x 8 cols = 40 units (matches your TSV data).
	// CA1: slightly larger than EC to allow flexible representations (60 units).
	// DG:  ~10x EC size for strong pattern separation (400 units, very sparse).
	// CA3: ~3x EC size for pattern completion via recurrence (120 units).
	in    := net.AddLayer2D("Input", 4, 8, leabra.InputLayer)
	ecin  := net.AddLayer2D("ECin",  4, 8, leabra.SuperLayer)
	ecout := net.AddLayer2D("ECout", 4, 8, leabra.TargetLayer)
	ca1   := net.AddLayer2D("CA1",   6, 10, leabra.SuperLayer)
	dg    := net.AddLayer2D("DG",    20, 20, leabra.SuperLayer)
	ca3   := net.AddLayer2D("CA3",   10, 12, leabra.SuperLayer)

	ecin.AddClass("EC")
	ecout.AddClass("EC")

	onetoone := paths.NewOneToOne()
	full     := paths.NewFull()

	net.ConnectLayers(in, ecin, onetoone, leabra.ForwardPath)
	net.ConnectLayers(ecout, ecin, onetoone, leabra.BackPath)

	// EC <-> CA1: full connectivity (no pools since layers are now 2D)
	net.ConnectLayers(ecin,  ca1,  full, leabra.EcCa1Path)
	net.ConnectLayers(ca1,   ecout, full, leabra.EcCa1Path)
	net.ConnectLayers(ecout, ca1,  full, leabra.EcCa1Path)

	ppath := paths.NewUniformRand()
	ppath.PCon = 0.25

	net.ConnectLayers(ecin, dg, ppath, leabra.CHLPath).AddClass("HippoCHL")
	net.ConnectLayers(ecin, ca3, ppath, leabra.EcCa1Path).AddClass("PPath")
	net.ConnectLayers(ca3, ca3, full, leabra.EcCa1Path).AddClass("PPath")

	mossy := paths.NewUniformRand()
	mossy.PCon = 0.02
	net.ConnectLayers(dg, ca3, mossy, leabra.CHLPath).AddClass("HippoCHL")

	net.ConnectLayers(ca3, ca1, full, leabra.CHLPath).AddClass("HippoCHL")

	ecin.PlaceRightOf(in, 2)
	ecout.PlaceRightOf(ecin, 2)
	dg.PlaceAbove(in)
	ca3.PlaceAbove(dg)
	ca1.PlaceRightOf(ca3, 2)

	net.Build()
	net.Defaults()
	ss.ApplyParams()
	net.InitWeights()
	net.InitTopoScales()
}

func (ss *Sim) ApplyParams() {
	ss.Params.Network = ss.Net
	ss.Params.SetAll()
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0))
	ss.Loops.ResetCounters()
	ss.GUI.StopNow = false
	ss.ApplyParams()
	ss.NewRun()
	ss.ViewUpdate.RecordSyns()
	ss.ViewUpdate.Update()
}

func (ss *Sim) TestInit() {
	tst := ss.Envs.ByMode(etime.Test).(*env.FixedTable)
	tst.Init(0)
}

// InitRandSeed initializes the random seed based on current training run number
func (ss *Sim) InitRandSeed(run int) {
	rand.Seed(ss.RandSeeds[run])
	ss.RandSeeds.Set(run)
	ss.RandSeeds.Set(run, &ss.Net.Rand)
	patgen.NewRand(ss.RandSeeds[run])
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	ls := looper.NewStacks()

	trls := ss.TrainAB.Rows
	ttrls := ss.TestAll.Rows

	ls.AddStack(etime.Train).AddTime(etime.Run, ss.Config.NRuns).AddTime(etime.Epoch, ss.Config.NEpochs).AddTime(etime.Trial, trls).AddTime(etime.Cycle, 100)
	ls.AddStack(etime.Test).AddTime(etime.Epoch, 1).AddTime(etime.Trial, ttrls).AddTime(etime.Cycle, 100)

	leabra.LooperStdPhases(ls, &ss.Context, ss.Net, 75, 99)
	leabra.LooperSimCycleAndLearn(ls, ss.Net, &ss.Context, &ss.ViewUpdate)
	ss.Net.ConfigLoopsHip(&ss.Context, ls)

	ls.Stacks[etime.Train].OnInit.Add("Init", func() { ss.Init() })
	ls.Stacks[etime.Test].OnInit.Add("Init", func() { ss.TestInit() })

	for m, _ := range ls.Stacks {
		stack := ls.Stacks[m]
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			ss.ApplyInputs()
		})
	}

	ls.Loop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	ls.Loop(etime.Train, etime.Run).OnEnd.Add("RunDone", func() {
		// Save results at end of each run, tagged with the overlap condition
		ss.SaveSummaryCSV("summary_results.csv")

		if ss.Stats.Int("Run") >= ss.Config.NRuns-1 {
			ss.RunStats()
			expt := ss.Stats.Int("Expt")
			ss.Stats.SetInt("Expt", expt+1)
		}
	})

	// Run a test pass at the end of every training epoch
	trainEpoch := ls.Loop(etime.Train, etime.Epoch)
	trainEpoch.OnEnd.Add("TestAtInterval", func() {
		if (ss.Config.TestInterval > 0) && ((trainEpoch.Counter.Cur+1)%ss.Config.TestInterval == 0) {
			ss.RunTestAll()
		}
	})

	// Early stop when ECoutAcc reaches StopMem threshold
	ls.Loop(etime.Train, etime.Epoch).IsDone.AddBool("ECoutAccStop", func() bool {
		tstEpcLog := ss.Logs.Tables[etime.Scope(etime.Test, etime.Epoch)]
		acc := float32(tstEpcLog.Table.Float("ECoutAcc", ss.Stats.Int("Epoch")))
		return acc >= ss.Config.StopMem
	})

	/////////////////////////////////////////////
	// Logging

	ls.Loop(etime.Test, etime.Epoch).OnEnd.Add("LogTestErrors", func() {
		leabra.LogTestErrors(&ss.Logs)
	})

	ls.AddOnEndToAll("Log", func(mode, time enums.Enum) {
		ss.Log(mode.(etime.Modes), time.(etime.Times))
	})
	leabra.LooperResetLogBelow(ls, &ss.Logs)

	leabra.LooperUpdateNetView(ls, &ss.ViewUpdate, ss.Net, ss.NetViewCounters)
	leabra.LooperUpdatePlots(ls, &ss.GUI)

	ls.Stacks[etime.Train].OnInit.Add("GUI-Init", func() { ss.GUI.UpdateWindow() })
	ls.Stacks[etime.Test].OnInit.Add("GUI-Init", func() { ss.GUI.UpdateWindow() })

	ss.Loops = ls
}

// ApplyInputs applies input patterns from given environment.
func (ss *Sim) ApplyInputs() {
	ctx := &ss.Context
	net := ss.Net
	ev := ss.Envs.ByMode(ctx.Mode).(*env.FixedTable)
	ecout := net.LayerByName("ECout")

	if ctx.Mode == etime.Train {
		ecout.Type = leabra.TargetLayer
	} else {
		ecout.Type = leabra.CompareLayer // don't clamp during test -- network must complete
	}
	ecout.UpdateExtFlags()
	net.InitExt()
	lays := net.LayersByType(leabra.InputLayer, leabra.TargetLayer)
	ev.Step()
	ss.Stats.SetString("TrialName", ev.TrialName.Cur)
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm)
		pats := ev.State(ly.Name)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

// NewRun initializes a new run of the model
func (ss *Sim) NewRun() {
	ctx := &ss.Context
	ss.InitRandSeed(ss.Loops.Loop(etime.Train, etime.Run).Counter.Cur)
	ss.ConfigEnv()
	ctx.Reset()
	ctx.Mode = etime.Train
	ss.Net.InitWeights()
	ss.InitStats()
	ss.StatCounters()
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
}

// TestAll runs through the full set of testing items
func (ss *Sim) RunTestAll() {
	ss.Envs.ByMode(etime.Test).Init(0)
	ss.Loops.ResetAndRun(etime.Test)
	ss.Loops.Mode = etime.Train
}

/////////////////////////////////////////////////////////////////////////
//   Patterns

// OpenPatAsset opens pattern file from embedded assets
func (ss *Sim) OpenPatAsset(dt *table.Table, fnm, name, desc string) error {
	dt.SetMetaData("name", name)
	dt.SetMetaData("desc", desc)
	err := dt.OpenFS(content, fnm, table.Tab)
	if errors.Log(err) == nil {
		for i := 1; i < dt.NumColumns(); i++ {
			dt.Columns[i].SetMetaData("grid-fill", "0.9")
		}
	}
	return err
}

func (ss *Sim) OpenPatterns() {
	fmt.Printf("Loading training patterns from: %s\n", ss.Config.PatternFile)
	err := ss.TrainAB.OpenCSV(core.Filename(ss.Config.PatternFile), table.Tab)
	if err != nil {
		panic(fmt.Sprintf("Failed to load training patterns from %s: %v", ss.Config.PatternFile, err))
	}
	ss.TrainAB.SetMetaData("name", fmt.Sprintf("Train_Overlap%d", ss.Config.OverlapPct))
	fmt.Printf("  Loaded %d training patterns\n", ss.TrainAB.Rows)

	if ss.Config.TestPatternFile != "" {
		fmt.Printf("Loading test patterns from: %s\n", ss.Config.TestPatternFile)
		err = ss.TestAll.OpenCSV(core.Filename(ss.Config.TestPatternFile), table.Tab)
		if err != nil {
			fmt.Printf("  WARNING: Could not load test patterns (%v).\n", err)
			fmt.Printf("  Falling back to training patterns as test set.\n")
			fmt.Printf("  NOTE: This means ECoutAcc measures reproduction, not recall.\n")
			ss.TestAll = ss.TrainAB.Clone()
		} else {
			fmt.Printf("  Loaded %d test patterns\n", ss.TestAll.Rows)
		}
	} else {
		fmt.Printf("No TestPatternFile set -- using training patterns as test set.\n")
		fmt.Printf("NOTE: ECoutAcc will measure reproduction, not recall.\n")
		ss.TestAll = ss.TrainAB.Clone()
	}
	ss.TestAll.SetMetaData("name", fmt.Sprintf("Test_Overlap%d", ss.Config.OverlapPct))
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
func (ss *Sim) InitStats() {
	ss.Stats.SetString("TrialName", "")
	ss.Stats.SetFloat("TrgOnWasOffAll", 0.0)
	ss.Stats.SetFloat("TrgOnWasOffCmp", 0.0) 
	ss.Stats.SetFloat("TrgOffWasOn", 0.0)    
	ss.Stats.SetFloat("ECoutAcc", 0.0)       
	ss.Stats.SetFloat("Mem", 0.0)            
	ss.Stats.SetInt("FirstPerfect", -1)

	ss.Logs.InitErrStats()
}

// StatCounters saves current counters to Stats
func (ss *Sim) StatCounters() {
	ctx := &ss.Context
	mode := ctx.Mode
	ss.Loops.Stacks[mode].CountersToStats(&ss.Stats)
	trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	ss.Stats.SetInt("Epoch", trnEpc)
	trl := ss.Stats.Int("Trial")
	ss.Stats.SetInt("Trial", trl)
	ss.Stats.SetInt("Cycle", int(ctx.Cycle))
	ss.Stats.SetString("TrialName", ss.Stats.String("TrialName"))
}

func (ss *Sim) NetViewCounters(tm etime.Times) {
	if ss.ViewUpdate.View == nil {
		return
	}
	if tm == etime.Trial {
		ss.TrialStats()
	}
	ss.StatCounters()
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "TrialName", "Cycle"})
}

func (ss *Sim) TrialStats() {
	ecout := ss.Net.LayerByName("ECout")

	actMi, _ := ecout.UnitVarIndex("ActM")
	targi, _ := ecout.UnitVarIndex("Targ")

	actThr := float32(0.5)
	correct := 0.0
	total := 0.0

	for i := 0; i < ecout.Shape.Len(); i++ {
		act := ecout.UnitValue1D(actMi, i, 0)
		trg := ecout.UnitValue1D(targi, i, 0)
		if (act > actThr && trg > actThr) || (act <= actThr && trg <= actThr) {
			correct++
		}
		total++
	}

	acc := 0.0
	if total > 0 {
		acc = correct / total
	}
	ss.Stats.SetFloat("ECoutAcc", acc)
}

func (ss *Sim) MemStats(mode etime.Modes) {
	memthr := 0.34
	ecout := ss.Net.LayerByName("ECout")
	inp := ss.Net.LayerByName("Input")
	nn := ecout.Shape.Len()
	actThr := float32(0.5)

	trgOnWasOffAll := 0.0
	trgOnWasOffCmp := 0.0 // completion failures
	trgOffWasOn := 0.0    // interference / false recall
	cmpN := 0.0
	trgOnN := 0.0
	trgOffN := 0.0

	actMi, _ := ecout.UnitVarIndex("ActM")
	targi, _ := ecout.UnitVarIndex("Targ")

	for ni := 0; ni < nn; ni++ {
		actm := ecout.UnitValue1D(actMi, ni, 0)
		trg := ecout.UnitValue1D(targi, ni, 0)
		inact := inp.UnitValue1D(actMi, ni, 0)

		if trg < actThr { // this unit should be OFF
			trgOffN++
			if actm > actThr {
				trgOffWasOn++ // network turned it ON -- interference error
			}
		} else { // this unit should be ON
			trgOnN++
			if inact < actThr { // ...and it wasn't in the input cue -- must recall from memory
				cmpN++
				if actm < actThr {
					trgOnWasOffAll++ // network failed to recall it
					trgOnWasOffCmp++ // specifically a completion failure
				}
			} else {
				if actm < actThr {
					trgOnWasOffAll++ // network failed even though cue provided it
				}
			}
		}
	}

	if trgOnN > 0 {
		trgOnWasOffAll /= trgOnN
	}
	if trgOffN > 0 {
		trgOffWasOn /= trgOffN
	}

	mem := 0.0
	if mode == etime.Train {
		// During training, no completion test -- just use ECoutAcc as proxy
		mem = ss.Stats.Float("ECoutAcc")
	} else {
		// During testing: score as remembered only if both error types are low
		if cmpN > 0 {
			trgOnWasOffCmp /= cmpN
		}
		if trgOnWasOffCmp < memthr && trgOffWasOn < memthr {
			mem = 1.0
		}
	}

	ss.Stats.SetFloat("Mem", mem)
	ss.Stats.SetFloat("TrgOnWasOffAll", trgOnWasOffAll)
	ss.Stats.SetFloat("TrgOnWasOffCmp", trgOnWasOffCmp)
	ss.Stats.SetFloat("TrgOffWasOn", trgOffWasOn)
}

func (ss *Sim) RunStats() {
	dt := ss.Logs.Table(etime.Train, etime.Run)
	runix := table.NewIndexView(dt)
	spl := split.GroupBy(runix, "Expt")
	split.DescColumn(spl, "TstMem")
	st := spl.AggsToTableCopy(table.AddAggName)
	ss.Logs.MiscTables["RunStats"] = st
	plt := ss.GUI.Plots[etime.ScopeKey("RunStats")]

	st.SetMetaData("XAxis", "RunName")
	st.SetMetaData("Points", "true")
	st.SetMetaData("TstMem:Mean:On", "+")
	st.SetMetaData("TstMem:Mean:FixMin", "true")
	st.SetMetaData("TstMem:Mean:FixMax", "true")
	st.SetMetaData("TstMem:Mean:Min", "0")
	st.SetMetaData("TstMem:Mean:Max", "1")
	st.SetMetaData("TstMem:Min:On", "+")
	st.SetMetaData("TstMem:Count:On", "-")

	plt.SetTable(st)
	plt.GoUpdatePlot()
}

func (ss *Sim) SaveSummaryCSV(filename string) {
	// Read from the TEST epoch log (not train) so we get genuine test-time metrics
	dt := ss.Logs.Table(etime.Test, etime.Epoch)
	if dt == nil || dt.Rows == 0 {
		fmt.Println("SaveSummaryCSV: no test epoch data available yet")
		return
	}

	last := dt.Rows - 1
	ecacc := dt.Float("ECoutAcc", last)
	cmpFail := dt.Float("TrgOnWasOffCmp", last) // completion failures
	interference := dt.Float("TrgOffWasOn", last) // interference errors
	mem := dt.Float("Mem", last)
	run := ss.Stats.Int("Run")

	fileExists := false
	if _, err := os.Stat(filename); err == nil {
		fileExists = true
	}

	f, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	if !fileExists {
		f.WriteString("OverlapPct,Run,ECoutAcc,TrgOnWasOffCmp,TrgOffWasOn,Mem\n")
	}

	line := fmt.Sprintf("%d,%d,%.4f,%.4f,%.4f,%.4f\n",
		ss.Config.OverlapPct, run, ecacc, cmpFail, interference, mem)
	f.WriteString(line)

	fmt.Printf("Saved results: Overlap=%d Run=%d ECoutAcc=%.3f CompletionFail=%.3f Interference=%.3f Mem=%.3f\n",
		ss.Config.OverlapPct, run, ecacc, cmpFail, interference, mem)
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) AddLogItems() {
	// These items copy test-epoch stats into the train-epoch log for easy plotting
	itemNames := []string{"TrgOnWasOffAll", "TrgOnWasOffCmp", "TrgOffWasOn", "Mem", "ECoutAcc"}
	for _, st := range itemNames {
		stnm := st
		tonm := "Tst" + st
		ss.Logs.AddItem(&elog.Item{
			Name: tonm,
			Type: reflect.Float64,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
					ctx.SetFloat64(ctx.ItemFloat(etime.Test, etime.Epoch, stnm))
				},
				etime.Scope(etime.Train, etime.Run): func(ctx *elog.Context) {
					ctx.SetFloat64(ctx.ItemFloat(etime.Test, etime.Epoch, stnm))
				}}})
	}
}

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0))

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddStatIntNoAggItem(etime.AllModes, etime.AllTimes, "Expt")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")

	// Core stats for the overlap study
	ss.Logs.AddStatAggItem("TrgOnWasOffAll", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("TrgOnWasOffCmp", etime.Run, etime.Epoch, etime.Trial) // completion failures
	ss.Logs.AddStatAggItem("TrgOffWasOn", etime.Run, etime.Epoch, etime.Trial)    // interference errors
	ss.Logs.AddStatAggItem("ECoutAcc", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("Mem", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatIntNoAggItem(etime.Train, etime.Run, "FirstPerfect")

	ss.AddLogItems()

	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	layers := ss.Net.LayersByType(leabra.SuperLayer, leabra.CTLayer, leabra.TargetLayer)
	leabra.LogAddDiagnosticItems(&ss.Logs, layers, etime.Train, etime.Epoch, etime.Trial)
	leabra.LogInputLayer(&ss.Logs, ss.Net, etime.Train)

	ss.Logs.AddLayerTensorItems(ss.Net, "ActM", etime.Test, etime.Trial, "TargetLayer")
	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Test, etime.Trial, "TargetLayer")

	// Plot the most informative metrics for the overlap study
	ss.Logs.PlotItems("ECoutAcc", "TrgOnWasOffCmp", "TrgOffWasOn")

	ss.Logs.CreateTables()

	// Train/Run level display settings
	ss.Logs.SetMeta(etime.Train, etime.Run, "TstMem:On", "+")
	ss.Logs.SetMeta(etime.Train, etime.Run, "TstECoutAcc:On", "+")
	ss.Logs.SetMeta(etime.Train, etime.Run, "TstTrgOnWasOffCmp:On", "+")
	ss.Logs.SetMeta(etime.Train, etime.Run, "TstTrgOffWasOn:On", "+")
	ss.Logs.SetMeta(etime.Train, etime.Run, "Type", "Bar")

	// Train/Epoch level display settings
	ss.Logs.SetMeta(etime.Train, etime.Epoch, "ECoutAcc:On", "+")
	ss.Logs.SetMeta(etime.Train, etime.Epoch, "TrgOnWasOffCmp:On", "+")
	ss.Logs.SetMeta(etime.Train, etime.Epoch, "TrgOffWasOn:On", "+")
	ss.Logs.SetMeta(etime.Train, etime.Epoch, "Mem:On", "+")
	ss.Logs.SetMeta(etime.Train, etime.Epoch, "TrgOnWasOffAll:On", "-")

	ss.Logs.SetContext(&ss.Stats, ss.Net)
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Run)
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	ctx := &ss.Context
	if mode != etime.Analyze {
		ctx.Mode = mode
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
		ss.MemStats(mode) // ← NOW CALLED: computes TrgOnWasOffCmp, TrgOffWasOn, Mem
		ss.StatCounters()
		ss.Logs.LogRow(mode, time, row)
		return
	}

	ss.Logs.LogRow(mode, time, row)
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "Hippocampus"
	ss.GUI.MakeBody(ss, "hip", title, `runs a hippocampus model on the AB-AC paired associate learning task. See <a href="https://github.com/compcogneuro/sims/blob/master/ch7/hip/README.md">README.md on GitHub</a>.</p>`, readme)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("Network")
	nv.Options.Raster.Max = 100
	nv.Options.MaxRecs = 300
	nv.SetNet(ss.Net)
	ss.ViewUpdate.Config(nv, etime.Phase, etime.Phase)
	ss.GUI.ViewUpdate = &ss.ViewUpdate

	ss.GUI.AddPlots(title, &ss.Logs)

	stnm := "RunStats"
	dt := ss.Logs.MiscTable(stnm)
	bcp, _ := ss.GUI.Tabs.NewTab(stnm + " Plot")
	plt := plotcore.NewSubPlot(bcp)
	ss.GUI.Plots[etime.ScopeKey(stnm)] = plt
	plt.Options.Title = "Run Stats"
	plt.Options.XAxis = "RunName"
	plt.SetTable(dt)

	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddLooperCtrl(p, ss.Loops)

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
			core.TheApp.OpenURL("https://github.com/compcogneuro/sims/blob/master/ch7/hip/README.md")
		},
	})
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
}

// Needed to satisfy interface -- unused in overlap study
var _ = math.NaN
var _ = strings.Contains
