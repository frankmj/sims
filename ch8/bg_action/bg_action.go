// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
bg_action implements a biologically realistic basal ganglia model for
the probabilistic selection (PS) task (Frank et al., 2004). Unlike the
simplified BG model in which actions are provided as input, this model
selects its own actions via disinhibitory gating of thalamus / premotor
cortex, with exploration driven by cortical noise.

Translated from the emergent 7 project: BG_4s_inhib_PS_e7a.proj
*/
package main

import (
	"bytes"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/etable/split"
	"github.com/emer/leabra/leabra"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
)

func main() {
	TheSim.New()
	TheSim.Config()
	if len(os.Args) > 1 {
		TheSim.CmdArgs() // simple abs-mode
		return
	}
	gimain.Main(func() { // this starts gui -- requiresass'n into var
		guirun()
	})
}

func guirun() {
	TheSim.Init()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}

// ParamSets defines the parameters for the simulation.
// The Base set provides the core parameters from the original emergent 7 project.
var ParamSets = params.Sets{
	{Name: "Base", Desc: "base parameters from BG_4s_inhib_PS_e7a.proj", Sheets: params.Sheets{
		"Network": &params.Sheet{
			// -- Input layer (clamped) --
			{Sel: "#Input", Desc: "input layer",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "2",
					"Layer.Act.XX1.Gain":   "600",
					"Layer.Act.XX1.Thr":    "0.25",
					"Layer.Act.Dt.VmTau":   "10",
					"Layer.Act.Dt.GTau":    "1.4",
				}},
			{Sel: "#Context", Desc: "context layer",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "2",
					"Layer.Act.XX1.Gain":   "600",
					"Layer.Act.XX1.Thr":    "0.25",
				}},
			// -- Matrisome (Go / NoGo) layers --
			{Sel: "#Go", Desc: "Go (D1R) MSN layer",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "2",
					"Layer.Inhib.Layer.FB": "0",
					"Layer.Act.XX1.Gain":   "600",
					"Layer.Act.XX1.Thr":    "0.26",
					"Layer.Act.Dt.VmTau":   "43.5",
					"Layer.Act.Dt.GTau":    "1.4",
					"Layer.Act.Gbar.E":     "1",
					"Layer.Act.Gbar.L":     "0.35",
					"Layer.Act.Gbar.I":     "7.5",
					"Layer.Act.Erev.E":     "1",
					"Layer.Act.Erev.L":     "0.15",
					"Layer.Act.Erev.I":     "0.15",
				}},
			{Sel: "#NoGo", Desc: "NoGo (D2R) MSN layer",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "2",
					"Layer.Inhib.Layer.FB": "0",
					"Layer.Act.XX1.Gain":   "600",
					"Layer.Act.XX1.Thr":    "0.26",
					"Layer.Act.Dt.VmTau":   "43.5",
					"Layer.Act.Dt.GTau":    "1.4",
					"Layer.Act.Gbar.E":     "1",
					"Layer.Act.Gbar.L":     "0.35",
					"Layer.Act.Gbar.I":     "7.5",
					"Layer.Act.Erev.E":     "1",
					"Layer.Act.Erev.L":     "0.15",
					"Layer.Act.Erev.I":     "0.15",
				}},
			// -- Striatum_Inhib layer --
			{Sel: "#StriatumInhib", Desc: "striatal inhibitory interneurons",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "2",
					"Layer.Act.XX1.Gain":   "600",
					"Layer.Act.XX1.Thr":    "0.25",
				}},
			// -- GPe layer --
			{Sel: "#GPe", Desc: "external segment of globus pallidus",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "2",
					"Layer.Inhib.Layer.FB": "0",
					"Layer.Act.XX1.Gain":   "600",
					"Layer.Act.XX1.Thr":    "0.25",
					"Layer.Act.Dt.VmTau":   "10",
					"Layer.Act.Dt.GTau":    "1.4",
					"Layer.Act.Gbar.E":     "1",
					"Layer.Act.Gbar.L":     "1",
					"Layer.Act.Gbar.I":     "1.5",
					"Layer.Act.Erev.E":     "1",
					"Layer.Act.Erev.L":     "0.26",
					"Layer.Act.Erev.I":     "0.15",
				}},
			// -- GPi layer --
			{Sel: "#GPi", Desc: "internal segment of globus pallidus",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "2",
					"Layer.Inhib.Layer.FB": "0",
					"Layer.Act.XX1.Gain":   "600",
					"Layer.Act.XX1.Thr":    "0.25",
					"Layer.Act.Dt.VmTau":   "10",
					"Layer.Act.Dt.GTau":    "1.4",
					"Layer.Act.Gbar.E":     "1",
					"Layer.Act.Gbar.L":     "1",
					"Layer.Act.Gbar.I":     "1.5",
					"Layer.Act.Erev.E":     "1",
					"Layer.Act.Erev.L":     "0.26",
					"Layer.Act.Erev.I":     "0.15",
				}},
			// -- Thalamus layer --
			{Sel: "#Thalamus", Desc: "thalamus",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "2",
					"Layer.Inhib.Layer.FB": "0",
					"Layer.Act.XX1.Gain":   "600",
					"Layer.Act.XX1.Thr":    "0.25",
					"Layer.Act.Dt.VmTau":   "20",
					"Layer.Act.Dt.GTau":    "2.5",
					"Layer.Act.Gbar.E":     "0.5",
					"Layer.Act.Gbar.L":     "0.07",
					"Layer.Act.Gbar.I":     "1.7",
					"Layer.Act.Erev.E":     "1",
					"Layer.Act.Erev.L":     "0.15",
					"Layer.Act.Erev.I":     "0.15",
				}},
			// -- PMC (Premotor Cortex) --
			{Sel: "#PMC", Desc: "premotor cortex",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "2",
					"Layer.Act.XX1.Gain":   "600",
					"Layer.Act.XX1.Thr":    "0.25",
					"Layer.Act.Dt.VmTau":   "20",
					"Layer.Act.Dt.GTau":    "1.4",
					"Layer.Act.Gbar.E":     "1",
					"Layer.Act.Gbar.L":     "0.1",
					"Layer.Act.Gbar.I":     "1",
					"Layer.Act.Erev.E":     "1",
					"Layer.Act.Erev.L":     "0.15",
					"Layer.Act.Erev.I":     "0.15",
					"Layer.Act.Noise.Dist":  "Gaussian",
					"Layer.Act.Noise.Var":   "0.001",
					"Layer.Act.Noise.Type":  "GeNoise",
					"Layer.Act.Noise.Fixed": "false",
				}},
			// -- Output layer --
			{Sel: "#Output", Desc: "output layer",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "2",
					"Layer.Act.XX1.Gain":   "20",
					"Layer.Act.XX1.Thr":    "0.2",
					"Layer.Act.Dt.VmTau":   "33",
					"Layer.Act.Dt.GTau":    "10",
				}},
			// -- SNc layer --
			{Sel: "#SNc", Desc: "dopamine neurons (clamped externally)",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "2",
					"Layer.Act.XX1.Gain":   "600",
					"Layer.Act.XX1.Thr":    "0.25",
				}},

			// ---- Projection parameters ----

			// Input -> Go (D1 learning)
			{Sel: "#InputToGo", Desc: "input to Go D1 pathway",
				Params: params.Params{
					"Prjn.WtScale.Abs": "0.6",
					"Prjn.Learn.Lrate": "0.05",
					"Prjn.Learn.Hebb":  "0.1",
				}},
			// Input -> NoGo (D2 learning)
			{Sel: "#InputToNoGo", Desc: "input to NoGo D2 pathway",
				Params: params.Params{
					"Prjn.WtScale.Abs": "0.65",
					"Prjn.Learn.Lrate": "0.05",
					"Prjn.Learn.Hebb":  "0.1",
				}},
			// Context -> Go
			{Sel: "#ContextToGo", Desc: "context to Go",
				Params: params.Params{
					"Prjn.WtScale.Abs": "0.6",
					"Prjn.WtScale.Rel": "1.25",
					"Prjn.Learn.Lrate": "0",
					"Prjn.Learn.Hebb":  "0.1",
				}},
			// Context -> NoGo
			{Sel: "#ContextToNoGo", Desc: "context to NoGo",
				Params: params.Params{
					"Prjn.WtScale.Abs": "0.6",
					"Prjn.WtScale.Rel": "1.25",
					"Prjn.Learn.Lrate": "0",
					"Prjn.Learn.Hebb":  "0.1",
				}},
			// SNc -> Go (D1 dopamine)
			{Sel: "#SNcToGo", Desc: "D1 dopamine to Go pathway",
				Params: params.Params{
					"Prjn.WtScale.Abs": "0.5",
					"Prjn.WtScale.Rel": "2",
					"Prjn.Learn.Lrate": "0",
				}},
			// SNc -> NoGo (D2 dopamine inhibitory)
			{Sel: "#SNcToNoGo", Desc: "D2 dopamine to NoGo pathway (inhibitory)",
				Params: params.Params{
					"Prjn.WtScale.Abs": "0.5",
					"Prjn.WtScale.Rel": "1.5",
					"Prjn.Learn.Lrate": "0",
					"Prjn.WtInit.Mean": "0.5",
				}},
			// PMC -> Go (motor cortex to striatum)
			{Sel: "#PMCToGo", Desc: "motor cortex to Go",
				Params: params.Params{
					"Prjn.WtScale.Abs": "1",
					"Prjn.WtScale.Rel": "1.5",
					"Prjn.Learn.Lrate": "0",
					"Prjn.Learn.Hebb":  "0.1",
				}},
			// PMC -> NoGo (motor cortex to NoGo)
			{Sel: "#PMCToNoGo", Desc: "motor cortex to NoGo",
				Params: params.Params{
					"Prjn.WtScale.Abs": "0.65",
					"Prjn.WtScale.Rel": "2",
					"Prjn.Learn.Lrate": "0",
					"Prjn.Learn.Hebb":  "0.1",
				}},
			// NoGo -> Go (lateral inhibition)
			{Sel: "#NoGoToGo", Desc: "NoGo inhibits Go",
				Params: params.Params{
					"Prjn.WtScale.Abs": "0.6",
					"Prjn.WtScale.Rel": "1.5",
					"Prjn.Learn.Lrate": "0",
				}},
			// NoGo -> GPe
			{Sel: "#NoGoToGPe", Desc: "NoGo inhibits GPe",
				Params: params.Params{
					"Prjn.WtScale.Abs": "0.4",
					"Prjn.WtScale.Rel": "1.2",
					"Prjn.Learn.Lrate": "0",
				}},
			// Go -> GPi (direct pathway inhibition)
			{Sel: "#GoToGPi", Desc: "Go inhibits GPi",
				Params: params.Params{
					"Prjn.WtScale.Abs": "1",
					"Prjn.WtScale.Rel": "3",
					"Prjn.Learn.Lrate": "0",
				}},
			// GPe -> GPi (indirect pathway inhibition)
			{Sel: "#GPeToGPi", Desc: "GPe inhibits GPi",
				Params: params.Params{
					"Prjn.WtScale.Abs": "0.4",
					"Prjn.WtScale.Rel": "1.2",
					"Prjn.Learn.Lrate": "0",
				}},
			// GPi -> Thalamus (inhibition)
			{Sel: "#GPiToThalamus", Desc: "GPi inhibits thalamus",
				Params: params.Params{
					"Prjn.WtScale.Abs": "1",
					"Prjn.WtScale.Rel": "3",
					"Prjn.Learn.Lrate": "0",
				}},
			// Thalamus -> PMC
			{Sel: "#ThalamusToPMC", Desc: "thalamus excites PMC",
				Params: params.Params{
					"Prjn.WtScale.Abs": "1",
					"Prjn.Learn.Lrate": "0",
				}},
			// PMC -> Thalamus (recurrent)
			{Sel: "#PMCToThalamus", Desc: "PMC recurrent to thalamus",
				Params: params.Params{
					"Prjn.WtScale.Abs": "0.7",
					"Prjn.WtScale.Rel": "5",
					"Prjn.Learn.Lrate": "0",
				}},
			// Input -> PMC (prepotent)
			{Sel: "#InputToPMC", Desc: "input to premotor cortex prepotent bias",
				Params: params.Params{
					"Prjn.WtScale.Abs": "5",
					"Prjn.Learn.Lrate": "0",
				}},
			// Context -> PMC
			{Sel: "#ContextToPMC", Desc: "context to premotor cortex",
				Params: params.Params{
					"Prjn.WtScale.Abs": "3",
					"Prjn.Learn.Lrate": "0",
				}},
			// PMC -> Output
			{Sel: "#PMCToOutput", Desc: "premotor cortex to output",
				Params: params.Params{
					"Prjn.WtScale.Abs": "1",
					"Prjn.Learn.Lrate": "0",
				}},
			// StriatumInhib projections
			{Sel: "#StriatumInhibToGo", Desc: "inhib interneurons to Go",
				Params: params.Params{
					"Prjn.WtScale.Abs": "0.3",
					"Prjn.WtScale.Rel": "2",
					"Prjn.Learn.Lrate": "0",
				}},
			{Sel: "#StriatumInhibToNoGo", Desc: "inhib interneurons to NoGo",
				Params: params.Params{
					"Prjn.WtScale.Abs": "0.3",
					"Prjn.WtScale.Rel": "2",
					"Prjn.Learn.Lrate": "0",
				}},
		},
	}},
	{Name: "PD", Desc: "Parkinson's Disease: reduced DA (2/4 intact SNc units)", Sheets: params.Sheets{
		"Sim": &params.Sheet{
			{Sel: "Sim", Desc: "reduce intact SNc units",
				Params: params.Params{
					"Sim.NumIntactSNc": "2",
				}},
		},
	}},
	{Name: "D2Agonist", Desc: "D2 agonist: increase D2 projection strength", Sheets: params.Sheets{
		"Sim": &params.Sheet{
			{Sel: "Sim", Desc: "double D2 strength",
				Params: params.Params{
					"Sim.D2Acq":  "0.15",
					"Sim.D2Perf": "0.15",
				}},
		},
	}},
}

// Sim encapsulates the entire simulation.
type Sim struct {

	// network
	Net *leabra.Network `view:"no-inline" desc:"the network"`

	// ---- task parameters ----

	// NumIntactSNc is the number of intact SNc (dopamine) units (out of 4).
	// Default 4 = healthy; 2 = simulated Parkinson's disease.
	NumIntactSNc int `desc:"number of intact SNc units (out of 4) -- 4=healthy, 2=PD"`

	// DA parameters
	D1Acq  float64 `desc:"D1 projection strength during acquisition"`
	D1Perf float64 `desc:"D1 projection strength during performance/test"`
	D2Acq  float64 `desc:"D2 projection strength during acquisition"`
	D2Perf float64 `desc:"D2 projection strength during performance/test"`

	// DA signal values
	DABurstVal float64 `desc:"SNc external value for DA burst (reward)"`
	DADipVal   float64 `desc:"SNc external value for DA dip (punishment)"`
	TonicDA    float64 `desc:"tonic DA level for neutral outcomes"`

	// PMC bias for candidate responses
	PMCBias float64 `desc:"bias weight applied to candidate PMC units"`

	// ---- training parameters ----
	MaxEpochs int `desc:"maximum number of training epochs"`
	MaxCycles int `desc:"maximum cycles per trial (settling)"`
	TestEpochs int `desc:"number of test epochs"`
	NTrialsPerEpoch int `desc:"number of trials per epoch"`

	// ---- noise schedule ----
	NoiseStart float64 `desc:"starting noise multiplier"`

	// ---- state ----
	TrainEnv   PSEnv       `desc:"training environment"`
	TestEnv    PSEnv       `desc:"testing environment"`
	Time       leabra.Time `desc:"time context"`
	IsRunning  bool        `view:"-" desc:"true when running"`
	StopNow    bool        `view:"-" desc:"flag to stop"`

	// ---- logging ----
	TrnEpcLog  *etable.Table `view:"no-inline" desc:"training epoch log"`
	TstEpcLog  *etable.Table `view:"no-inline" desc:"testing epoch log"`
	TrnTrlLog  *etable.Table `view:"no-inline" desc:"training trial log"`
	TstTrlLog  *etable.Table `view:"no-inline" desc:"testing trial log"`
	RunLog     *etable.Table `view:"no-inline" desc:"run log"`
	RunStats   *etable.Table `view:"no-inline" desc:"run stats summary"`

	// ---- plots ----
	TrnEpcPlot  *eplot.Plot2D `view:"-" desc:"training epoch plot"`
	TstEpcPlot  *eplot.Plot2D `view:"-" desc:"test epoch plot"`
	TrnTrlPlot  *eplot.Plot2D `view:"-" desc:"training trial plot"`
	TstTrlPlot  *eplot.Plot2D `view:"-" desc:"testing trial plot"`
	RunPlot     *eplot.Plot2D `view:"-" desc:"run plot"`

	// ---- gui ----
	NetView   *netview.NetView `view:"-" desc:"net view"`
	Win       *gi.Window       `view:"-" desc:"main gui window"`
	ToolBar   *gi.ToolBar      `view:"-" desc:"toolbar"`
	ValsTsr   *etensor.Float32 `view:"-" desc:"temp tensor for vals"`

	// ---- current trial state ----
	TrialName string  `view:"-" desc:"current trial name"`
	Action    int     `view:"-" desc:"selected action (1-4, 0=no action)"`
	Gated     bool    `view:"-" desc:"whether an action was gated by BG"`
	Rewarded  bool    `view:"-" desc:"whether reward was delivered"`
	DASignal  float64 `view:"-" desc:"DA signal applied on this trial"`

	// ---- counters ----
	Epoch   int `view:"-" desc:"current epoch"`
	Trial   int `view:"-" desc:"current trial"`
	Run     int `view:"-" desc:"current run"`
	NRuns   int `desc:"total number of runs"`
	Testing bool `view:"-" desc:"true during test phase"`
}

// TheSim is the overall state for this simulation
var TheSim Sim

// KiT_Sim provides type information for Sim
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// New creates a new Sim with default parameters.
func (ss *Sim) New() {
	ss.Net = &leabra.Network{}
	ss.NumIntactSNc = 4
	ss.D1Acq = 0.6
	ss.D1Perf = 0.6
	ss.D2Acq = 0.075
	ss.D2Perf = 0.075
	ss.DABurstVal = 1.0
	ss.DADipVal = 0.0
	ss.TonicDA = 0.026
	ss.PMCBias = 3.0
	ss.MaxEpochs = 30
	ss.MaxCycles = 150
	ss.TestEpochs = 30
	ss.NTrialsPerEpoch = 100
	ss.NRuns = 50
	ss.NoiseStart = 1.0

	ss.TrnEpcLog = &etable.Table{}
	ss.TstEpcLog = &etable.Table{}
	ss.TrnTrlLog = &etable.Table{}
	ss.TstTrlLog = &etable.Table{}
	ss.RunLog = &etable.Table{}
	ss.RunStats = &etable.Table{}
	ss.ValsTsr = &etensor.Float32{}
}

// Config configures all elements of the simulation.
func (ss *Sim) Config() {
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
}

////////////////////////////////////////////////////////////////////
//  Environment

// PSEnv implements the Probabilistic Selection task environment.
type PSEnv struct {
	Nm       string          `desc:"name of this environment"`
	TrialCt  int             `desc:"current trial counter"`
	EpochCt  int             `desc:"current epoch counter"`
	TrialName string         `desc:"name of the current trial type"`
	TrialType int            `desc:"0 = 8020_R1R2, 1 = 6040_R3R4"`
	Order    []int           `view:"-" desc:"shuffled trial order"`
}

// Name returns the name of the environment
func (ev *PSEnv) Name() string { return ev.Nm }

// Init initializes the environment for a new run
func (ev *PSEnv) Init(nTrials int) {
	ev.TrialCt = 0
	ev.EpochCt = 0
	ev.Order = make([]int, nTrials)
	for i := range ev.Order {
		ev.Order[i] = i % 2 // alternating 8020 and 6040
	}
	ev.Shuffle()
}

// Shuffle randomizes the trial order
func (ev *PSEnv) Shuffle() {
	for i := len(ev.Order) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		ev.Order[i], ev.Order[j] = ev.Order[j], ev.Order[i]
	}
}

// Step advances to the next trial
func (ev *PSEnv) Step() {
	if ev.TrialCt >= len(ev.Order) {
		ev.TrialCt = 0
		ev.EpochCt++
		ev.Shuffle()
	}
	ev.TrialType = ev.Order[ev.TrialCt]
	if ev.TrialType == 0 {
		ev.TrialName = "8020_R1R2"
	} else {
		ev.TrialName = "6040_R3R4"
	}
	ev.TrialCt++
}

// ConfigEnv configures the training and testing environments.
func (ss *Sim) ConfigEnv() {
	ss.TrainEnv.Nm = "TrainEnv"
	ss.TrainEnv.Init(ss.NTrialsPerEpoch)
	ss.TestEnv.Nm = "TestEnv"
	ss.TestEnv.Init(ss.NTrialsPerEpoch)
}

////////////////////////////////////////////////////////////////////
//  Network

// ConfigNet builds the network according to BG_4s_inhib_PS_e7a.proj
func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.InitName(net, "BG_PS")

	// ---- Create layers ----
	// Input: 6x3 (18 units: first 6 = S1, next 12 = S2/shared)
	inp := net.AddLayer2D("Input", 6, 3, emer.Input)
	// Context: 9x2 (18 units)
	ctx := net.AddLayer2D("Context", 9, 2, emer.Input)
	// Go (D1R MSN): 6x6 (36 units, 4 action pools of 9)
	goLy := net.AddLayer2D("Go", 6, 6, emer.Hidden)
	// NoGo (D2R MSN): 6x6 (36 units)
	nogoLy := net.AddLayer2D("NoGo", 6, 6, emer.Hidden)
	// Striatum Inhib: 4x4 (16 units)
	strInhib := net.AddLayer2D("StriatumInhib", 4, 4, emer.Hidden)
	// GPe: 2x2 (4 units)
	gpe := net.AddLayer2D("GPe", 2, 2, emer.Hidden)
	// GPi: 4x2 (8 units)
	gpi := net.AddLayer2D("GPi", 4, 2, emer.Hidden)
	// Thalamus: 2x2 (4 units)
	thal := net.AddLayer2D("Thalamus", 2, 2, emer.Hidden)
	// PMC: 4x2 (8 units: 2 per action channel)
	pmc := net.AddLayer2D("PMC", 4, 2, emer.Hidden)
	// Output: 4x1 (4 units)
	out := net.AddLayer2D("Output", 4, 1, emer.Compare)
	// SNc: 2x2 (4 units -- externally clamped)
	snc := net.AddLayer2D("SNc", 2, 2, emer.Input)

	// ---- Create projections ----
	full := prjn.NewFull()
	_ = full

	// Input -> Go (D1 learning pathway)
	net.ConnectLayers(inp, goLy, full, emer.Forward).SetClass("InputToGo")
	// Input -> NoGo (D2 learning pathway)
	net.ConnectLayers(inp, nogoLy, full, emer.Forward).SetClass("InputToNoGo")
	// Context -> Go
	net.ConnectLayers(ctx, goLy, full, emer.Forward).SetClass("ContextToGo")
	// Context -> NoGo
	net.ConnectLayers(ctx, nogoLy, full, emer.Forward).SetClass("ContextToNoGo")
	// SNc -> Go (D1 dopamine modulation)
	net.ConnectLayers(snc, goLy, full, emer.Forward).SetClass("SNcToGo")
	// SNc -> NoGo (D2 dopamine modulation -- inhibitory)
	net.ConnectLayers(snc, nogoLy, full, emer.Inhib).SetClass("SNcToNoGo")
	// PMC -> Go (motor cortex feedback to Go)
	net.ConnectLayers(pmc, goLy, full, emer.Forward).SetClass("PMCToGo")
	// PMC -> NoGo (motor cortex feedback to NoGo)
	net.ConnectLayers(pmc, nogoLy, full, emer.Forward).SetClass("PMCToNoGo")
	// NoGo -> Go (lateral inhibition)
	net.ConnectLayers(nogoLy, goLy, full, emer.Inhib).SetClass("NoGoToGo")
	// StriatumInhib -> Go
	net.ConnectLayers(strInhib, goLy, full, emer.Inhib).SetClass("StriatumInhibToGo")
	// StriatumInhib -> NoGo
	net.ConnectLayers(strInhib, nogoLy, full, emer.Inhib).SetClass("StriatumInhibToNoGo")

	// Input -> StriatumInhib
	net.ConnectLayers(inp, strInhib, full, emer.Forward).SetClass("FFtoInhib")
	// Context -> StriatumInhib
	net.ConnectLayers(ctx, strInhib, full, emer.Forward).SetClass("FFtoInhib")
	// Go -> StriatumInhib (feedback)
	net.ConnectLayers(goLy, strInhib, full, emer.Forward).SetClass("FBtoInhib")
	// NoGo -> StriatumInhib (feedback)
	net.ConnectLayers(nogoLy, strInhib, full, emer.Forward).SetClass("FBtoInhib")
	// SNc -> StriatumInhib
	net.ConnectLayers(snc, strInhib, full, emer.Forward).SetClass("SNcToGo")
	// PMC -> StriatumInhib
	net.ConnectLayers(pmc, strInhib, full, emer.Forward).SetClass("PMCToGo")
	// StriatumInhib self-inhibition
	net.ConnectLayers(strInhib, strInhib, full, emer.Inhib).SetClass("InhibInhib")

	// NoGo -> GPe (inhibitory)
	net.ConnectLayers(nogoLy, gpe, full, emer.Inhib).SetClass("NoGoToGPe")
	// Go -> GPi (direct pathway, inhibitory)
	net.ConnectLayers(goLy, gpi, full, emer.Inhib).SetClass("GoToGPi")
	// GPe -> GPi (indirect pathway, inhibitory)
	net.ConnectLayers(gpe, gpi, full, emer.Inhib).SetClass("GPeToGPi")
	// GPi -> Thalamus (inhibitory)
	net.ConnectLayers(gpi, thal, full, emer.Inhib).SetClass("GPiToThalamus")
	// Thalamus -> PMC
	net.ConnectLayers(thal, pmc, full, emer.Forward).SetClass("ThalamusToPMC")
	// PMC -> Thalamus (recurrent)
	net.ConnectLayers(pmc, thal, full, emer.Forward).SetClass("PMCToThalamus")
	// Input -> PMC (prepotent response bias)
	net.ConnectLayers(inp, pmc, full, emer.Forward).SetClass("InputToPMC")
	// Context -> PMC
	net.ConnectLayers(ctx, pmc, full, emer.Forward).SetClass("ContextToPMC")
	// PMC -> Output
	net.ConnectLayers(pmc, out, full, emer.Forward).SetClass("PMCToOutput")

	// ---- Layout ----
	// Layer positions set by default layout

	net.Defaults()
	ss.SetParams("Network", false)
	net.Build()
	net.InitWts()
}

////////////////////////////////////////////////////////////////////
//  Params

// SetParams sets parameters from the ParamSets
func (ss *Sim) SetParams(sheet string, setMsg bool) {
	for _, ps := range ParamSets {
		if ps.Name != "Base" {
			continue
		}
		sh, ok := ps.Sheets[sheet]
		if !ok {
			continue
		}
		sh.Apply(ss.Net, setMsg)
	}
}

////////////////////////////////////////////////////////////////////
//  Init

// Init restarts the simulation.
func (ss *Sim) Init() {
	rand.Seed(time.Now().UnixNano())
	ss.ConfigEnv()
	ss.Time.Reset()
	ss.Net.InitWts()
	ss.Epoch = 0
	ss.Trial = 0
	ss.StopNow = false
	ss.UpdateView(true)
}

// NewRndSeed sets a new random seed
func (ss *Sim) NewRndSeed() {
	rand.Seed(time.Now().UnixNano())
}

////////////////////////////////////////////////////////////////////
//  Running

// Stop stops the simulation
func (ss *Sim) Stop() {
	ss.StopNow = true
}

// ApplyInputs applies the stimulus pattern for the current trial.
func (ss *Sim) ApplyInputs() {
	net := ss.Net
	var ev *PSEnv
	if ss.Testing {
		ev = &ss.TestEnv
	} else {
		ev = &ss.TrainEnv
	}
	ss.TrialName = ev.TrialName

	// Input layer: 6x3 = 18 units
	inp := net.LayerByName("Input").(*leabra.Layer)
	ctx := net.LayerByName("Context").(*leabra.Layer)
	snc := net.LayerByName("SNc").(*leabra.Layer)
	pmc := net.LayerByName("PMC").(*leabra.Layer)

	// Clear all inputs
	inp.SetType(emer.Input)
	ctx.SetType(emer.Input)
	snc.SetType(emer.Input)

	// Input patterns from the old proj file:
	// 8020_R1R2: Input=[0,0,0,0,0,0, 1,1,1,1,1,1, x,x,x,x,x,x] -- S1 in second row
	// 6040_R3R4: Input=[1,1,1,1,1,1, x,x,x,x,x,x, 0,0,0,0,0,0] -- S2 in first row
	inpPat := make([]float32, 18)
	ctxPat := make([]float32, 18)

	if ev.TrialType == 0 { // 8020_R1R2
		// S1: units 6-11 active (second block of 6)
		for i := 6; i < 12; i++ {
			inpPat[i] = 1.0
		}
		// Context for S1: first 3 of 18
		for i := 0; i < 6; i++ {
			ctxPat[i] = 1.0
		}
	} else { // 6040_R3R4
		// S2: units 12-17 active (third block of 6) -- note: the proj file has different layout
		// Actually from proj: row0=[1,1,1,1,1,1, 0,0,0,0,0,0] for 6040
		for i := 0; i < 6; i++ {
			inpPat[i] = 1.0
		}
		for i := 6; i < 12; i++ {
			ctxPat[i] = 1.0
		}
	}

	// Apply Input pattern
	inpTsr := etensor.NewFloat32([]int{6, 3}, nil, nil)
	copy(inpTsr.Values, inpPat)
	inp.ApplyExt(inpTsr)

	// Apply Context pattern
	ctxTsr := etensor.NewFloat32([]int{9, 2}, nil, nil)
	copy(ctxTsr.Values, ctxPat)
	ctx.ApplyExt(ctxTsr)

	// Apply tonic DA to SNc at start of minus phase
	sncTsr := etensor.NewFloat32([]int{2, 2}, nil, nil)
	for i := 0; i < 4; i++ {
		sncTsr.Values[i] = float32(ss.TonicDA)
	}
	snc.ApplyExt(sncTsr)

	// Apply PMC bias: bias the two candidate responses for this trial
	// 8020: bias units 0,1,4,5 (R1,R2 channels -- pairs of PMC units per action)
	// 6040: bias units 2,3,6,7 (R3,R4 channels)
	// First clear all PMC bias weights
	for i := 0; i < 8; i++ {
		nrn := &pmc.Neurons[i]
		_ = nrn
	}

	// Store trial info
	ss.Action = 0
	ss.Gated = false
	ss.Rewarded = false
	ss.DASignal = ss.TonicDA
}

// ApplyPMCBias applies bias weights to candidate PMC units.
// In the original model this was done via bias weight manipulation.
// Here we approximate by adding extra excitatory input to candidate units.
func (ss *Sim) ApplyPMCBias() {
	pmc := ss.Net.LayerByName("PMC").(*leabra.Layer)
	var ev *PSEnv
	if ss.Testing {
		ev = &ss.TestEnv
	} else {
		ev = &ss.TrainEnv
	}
	if ev.TrialType == 0 { // 8020_R1R2: R1(units 0,4) and R2(units 1,5) are candidates
		pmc.Neurons[0].Ext = float32(ss.PMCBias)
		pmc.Neurons[1].Ext = float32(ss.PMCBias)
		pmc.Neurons[4].Ext = float32(ss.PMCBias)
		pmc.Neurons[5].Ext = float32(ss.PMCBias)
	} else { // 6040_R3R4: R3(units 2,6) and R4(units 3,7)
		pmc.Neurons[2].Ext = float32(ss.PMCBias)
		pmc.Neurons[3].Ext = float32(ss.PMCBias)
		pmc.Neurons[6].Ext = float32(ss.PMCBias)
		pmc.Neurons[7].Ext = float32(ss.PMCBias)
	}
}

// DetermineAction reads PMC minus-phase activations to determine selected action.
// Action selection is based on summed activation of paired PMC units.
// PMC has 8 units: pairs (0,4)=R1, (1,5)=R2, (2,6)=R3, (3,7)=R4
func (ss *Sim) DetermineAction() {
	pmc := ss.Net.LayerByName("PMC").(*leabra.Layer)
	out := ss.Net.LayerByName("Output").(*leabra.Layer)

	// Read minus-phase activations from PMC
	actM := make([]float64, 8)
	for i := 0; i < 8; i++ {
		actM[i] = float64(pmc.Neurons[i].ActM)
	}

	// Sum paired units for each action
	actionActs := [4]float64{
		actM[0] + actM[4], // R1
		actM[1] + actM[5], // R2
		actM[2] + actM[6], // R3
		actM[3] + actM[7], // R4
	}

	// Check if Output layer has any activation (indicates gating occurred)
	maxOut := float32(0)
	for i := 0; i < len(out.Neurons); i++ {
		if out.Neurons[i].Act > maxOut {
			maxOut = out.Neurons[i].Act
		}
	}

	if maxOut > 0 {
		// Find the winning action
		maxAct := actionActs[0]
		maxIdx := 0
		for i := 1; i < 4; i++ {
			if actionActs[i] > maxAct {
				maxAct = actionActs[i]
				maxIdx = i
			}
		}
		ss.Action = maxIdx + 1 // 1-4
		ss.Gated = true
	} else {
		// No gating -- response determined purely by noise/cortical dynamics
		// Still pick the max as the "selected" action
		maxAct := actionActs[0]
		maxIdx := 0
		for i := 1; i < 4; i++ {
			if actionActs[i] > maxAct {
				maxAct = actionActs[i]
				maxIdx = i
			}
		}
		ss.Action = maxIdx + 1
		ss.Gated = false
	}

	// Clamp the winning action into PMC for plus phase (bias weight = 10)
	// This ensures plus-phase reflects the chosen action
	for i := 0; i < 8; i++ {
		pmc.Neurons[i].Ext = 0
	}
	winIdx := ss.Action - 1
	pmc.Neurons[winIdx].Ext = 10
	pmc.Neurons[winIdx+4].Ext = 10
}

// DeliverReward determines reward based on selected action and trial type,
// then sets SNc DA values accordingly.
func (ss *Sim) DeliverReward() {
	var ev *PSEnv
	if ss.Testing {
		ev = &ss.TestEnv
	} else {
		ev = &ss.TrainEnv
	}

	snc := ss.Net.LayerByName("SNc").(*leabra.Layer)
	k := float64(ss.NumIntactSNc) / 4.0

	var rewarded bool
	var daVal float64

	if strings.Contains(ev.TrialName, "8020") {
		switch ss.Action {
		case 1: // R1: 80% reward
			if rand.Intn(10) > 1 { // >1 means 80% chance
				rewarded = true
			}
		case 2: // R2: 20% reward (80% punishment)
			if rand.Intn(10) > 7 { // >7 means 20% chance
				rewarded = true
			}
		default: // R3/R4 during 8020 = neutral
			daVal = ss.TonicDA
		}
	} else if strings.Contains(ev.TrialName, "6040") {
		switch ss.Action {
		case 3: // R3: 60% reward
			if rand.Intn(10) > 3 { // >3 means 60% chance (70%? -- proj has >3 = 70%)
				rewarded = true
			}
		case 4: // R4: 40% reward
			if rand.Intn(10) > 5 { // >5 means 40% chance
				rewarded = true
			}
		default: // R1/R2 during 6040 = neutral
			daVal = ss.TonicDA
		}
	}

	if ss.Action >= 1 && ss.Action <= 4 {
		if ev.TrialType == 0 && (ss.Action == 1 || ss.Action == 2) {
			if rewarded {
				daVal = ss.DABurstVal
			} else {
				daVal = ss.DADipVal
			}
		} else if ev.TrialType == 1 && (ss.Action == 3 || ss.Action == 4) {
			if rewarded {
				daVal = ss.DABurstVal
			} else {
				daVal = ss.DADipVal
			}
		} else {
			daVal = ss.TonicDA
		}
	}

	ss.Rewarded = rewarded
	ss.DASignal = daVal

	// Apply DA to SNc units
	nSNc := len(snc.Neurons)
	sncTsr := etensor.NewFloat32([]int{2, 2}, nil, nil)
	for i := 0; i < nSNc; i++ {
		if i < int(k*float64(nSNc)) {
			sncTsr.Values[i] = float32(daVal)
		} else {
			sncTsr.Values[i] = 0 // damaged units
		}
	}
	snc.ApplyExt(sncTsr)
}

// NoiseMultiplier returns the noise multiplier for the current epoch
// based on the noise schedule from the original model.
// Schedule: epoch 0-25: 1.0, 25-80: ramp 1.0->0.5, 80-100: ramp 0.5->0.2
func (ss *Sim) NoiseMultiplier() float64 {
	ep := ss.Epoch
	if ep <= 25 {
		return 1.0
	} else if ep <= 80 {
		return 1.0 - float64(ep-25)*0.5/55.0
	} else if ep <= 100 {
		return 0.5 - float64(ep-80)*0.3/20.0
	}
	return 0.2
}

// Counters returns a string of current counter values for display.
func (ss *Sim) Counters() string {
	return fmt.Sprintf("Epoch:\t%d\tTrial:\t%d\tCycle:\t%d\tAction:\t%d", ss.Epoch, ss.Trial, ss.Time.Cycle, ss.Action)
}

// AlphaCyc runs one alpha-cycle (trial) of settling.
// Uses the standard quarter-based leabra settling:
// Q1-Q3 = minus phase, Q4 = plus phase.
// After minus phase (end of Q3), we determine action and deliver DA.
func (ss *Sim) AlphaCyc() {
	net := ss.Net

	// Update prior weight changes at start
	if !ss.Testing {
		net.WtFmDWt()
	}

	net.AlphaCycInit(true) // true = learning
	ss.Time.AlphaCycStart()

	// Run all 4 quarters
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			net.Cycle(&ss.Time)
			ss.Time.CycleInc()
		}

		// After Q3 (minus phase complete), determine action and set DA for plus phase
		if qtr == 2 {
			ss.DetermineAction()
			if !ss.Testing {
				ss.DeliverReward()
			} else {
				snc := net.LayerByName("SNc").(*leabra.Layer)
				sncTsr := etensor.NewFloat32([]int{2, 2}, nil, nil)
				for i := 0; i < 4; i++ {
					sncTsr.Values[i] = float32(ss.TonicDA)
				}
				snc.ApplyExt(sncTsr)
				ss.DASignal = ss.TonicDA
				ss.Rewarded = false
			}
		}

		net.QuarterFinal(&ss.Time)
		ss.Time.QuarterInc()
	}

	// Learning (only during training)
	if !ss.Testing {
		net.DWt()
		// Apply DA modulation to learning:
		// Go pathway weights are modulated by DA (D1 rule)
		// NoGo pathway weights are modulated by -DA (D2 rule)
		ss.ApplyDAModulation()
	}

	ss.UpdateView(!ss.Testing)
}

// ApplyDAModulation modulates the weight changes in Go and NoGo pathways
// based on the DA signal, implementing the D1/D2 learning rule.
func (ss *Sim) ApplyDAModulation() {
	net := ss.Net
	da := ss.DASignal - ss.TonicDA // phasic DA = total - tonic

	goLy := net.LayerByName("Go").(*leabra.Layer)
	nogoLy := net.LayerByName("NoGo").(*leabra.Layer)

	// Modulate Go (D1) pathway: positive DA -> more learning
	for pi := 0; pi < goLy.NRecvPrjns(); pi++ {
		pj := goLy.RecvPrjn(pi).(*leabra.Prjn)
		if pj.Off {
			continue
		}
		clNm := pj.Cls
		if strings.Contains(clNm, "Input") || strings.Contains(clNm, "Context") {
			// DA-modulated learning pathways
			for si := range pj.Syns {
				if da > 0 {
					pj.Syns[si].DWt *= float32(1.0 + da)
				} else {
					pj.Syns[si].DWt *= float32(math.Max(0, 1.0+da))
				}
			}
		}
	}

	// Modulate NoGo (D2) pathway: negative DA -> more learning (reversed)
	for pi := 0; pi < nogoLy.NRecvPrjns(); pi++ {
		pj := nogoLy.RecvPrjn(pi).(*leabra.Prjn)
		if pj.Off {
			continue
		}
		clNm := pj.Cls
		if strings.Contains(clNm, "Input") || strings.Contains(clNm, "Context") {
			for si := range pj.Syns {
				if da < 0 {
					pj.Syns[si].DWt *= float32(1.0 + math.Abs(da))
				} else {
					pj.Syns[si].DWt *= float32(math.Max(0, 1.0-da))
				}
			}
		}
	}
}

// TrainTrial runs one training trial.
func (ss *Sim) TrainTrial() {
	ss.Testing = false
	ss.TrainEnv.Step()
	ss.TrialName = ss.TrainEnv.TrialName
	ss.ApplyInputs()
	ss.ApplyPMCBias()
	ss.AlphaCyc()
	ss.LogTrnTrl(ss.TrnTrlLog)
}

// TrainEpoch runs one training epoch.
func (ss *Sim) TrainEpoch() {
	for trl := 0; trl < ss.NTrialsPerEpoch; trl++ {
		ss.Trial = trl
		ss.TrainTrial()
		if ss.StopNow {
			return
		}
	}
	ss.LogTrnEpc(ss.TrnEpcLog)
	ss.Epoch++
}

// Train runs the full training procedure.
func (ss *Sim) Train() {
	ss.IsRunning = true
	ss.StopNow = false
	// Set D1/D2 strength for acquisition
	ss.SetDAStrength(ss.D1Acq, ss.D2Acq)
	for ep := 0; ep < ss.MaxEpochs; ep++ {
		ss.TrainEpoch()
		if ss.StopNow {
			break
		}
	}
	ss.IsRunning = false
	ss.UpdateView(false)
}

// TestTrial runs one test trial.
func (ss *Sim) TestTrial() {
	ss.Testing = true
	ss.TestEnv.Step()
	ss.TrialName = ss.TestEnv.TrialName
	ss.ApplyInputs()
	ss.ApplyPMCBias()
	ss.AlphaCyc()
	ss.LogTstTrl(ss.TstTrlLog)
}

// TestEpoch runs one test epoch.
func (ss *Sim) TestEpoch() {
	for trl := 0; trl < ss.NTrialsPerEpoch; trl++ {
		ss.Trial = trl
		ss.TestTrial()
		if ss.StopNow {
			return
		}
	}
	ss.LogTstEpc(ss.TstEpcLog)
}

// Test runs the test phase.
func (ss *Sim) Test() {
	ss.IsRunning = true
	ss.StopNow = false
	// Set D1/D2 strength for performance
	ss.SetDAStrength(ss.D1Perf, ss.D2Perf)
	for ep := 0; ep < ss.TestEpochs; ep++ {
		ss.TestEpoch()
		if ss.StopNow {
			break
		}
	}
	ss.IsRunning = false
	ss.UpdateView(false)
}

// TrainAndTest runs the full train+test cycle.
func (ss *Sim) TrainAndTest() {
	ss.Train()
	if !ss.StopNow {
		ss.Test()
	}
}

// RunBatch runs multiple networks (for batch statistics).
func (ss *Sim) RunBatch() {
	ss.IsRunning = true
	ss.StopNow = false
	for run := 0; run < ss.NRuns; run++ {
		ss.Run = run
		ss.Init()
		ss.TrainAndTest()
		ss.LogRun(ss.RunLog)
		if ss.StopNow {
			break
		}
	}
	ss.IsRunning = false
	ss.UpdateView(false)
}

// SetDAStrength sets the D1 and D2 projection strengths.
func (ss *Sim) SetDAStrength(d1, d2 float64) {
	net := ss.Net
	goLy := net.LayerByName("Go").(*leabra.Layer)
	nogoLy := net.LayerByName("NoGo").(*leabra.Layer)

	// Set SNc->Go (D1) wt_scale.abs
	for pi := 0; pi < goLy.NRecvPrjns(); pi++ {
		pj := goLy.RecvPrjn(pi).(*leabra.Prjn)
		if strings.Contains(pj.Cls, "SNcToGo") {
			pj.WtScale.Abs = float32(d1)
		}
	}
	// Set SNc->NoGo (D2) wt_scale.abs
	for pi := 0; pi < nogoLy.NRecvPrjns(); pi++ {
		pj := nogoLy.RecvPrjn(pi).(*leabra.Prjn)
		if strings.Contains(pj.Cls, "SNcToNoGo") {
			pj.WtScale.Abs = float32(d2)
		}
	}
}

////////////////////////////////////////////////////////////////////
//  Logging

// ConfigLogs sets up all log tables.
func (ss *Sim) ConfigLogs() {
	ss.ConfigTrnTrlLog(ss.TrnTrlLog)
	ss.ConfigTrnEpcLog(ss.TrnEpcLog)
	ss.ConfigTstTrlLog(ss.TstTrlLog)
	ss.ConfigTstEpcLog(ss.TstEpcLog)
	ss.ConfigRunLog(ss.RunLog)
}

// ConfigTrnTrlLog configures the training trial log.
func (ss *Sim) ConfigTrnTrlLog(dt *etable.Table) {
	dt.SetMetaData("name", "TrnTrlLog")
	dt.SetMetaData("desc", "Record of each training trial")
	dt.SetMetaData("read-only", "true")
	sch := etable.Schema{
		{"Epoch", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"Action", etensor.INT64, nil, nil},
		{"Gated", etensor.BOOL, nil, nil},
		{"Rewarded", etensor.BOOL, nil, nil},
		{"DA", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

// LogTrnTrl logs one training trial.
func (ss *Sim) LogTrnTrl(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)
	dt.SetCellFloat("Epoch", row, float64(ss.Epoch))
	dt.SetCellFloat("Trial", row, float64(ss.Trial))
	dt.SetCellString("TrialName", row, ss.TrialName)
	dt.SetCellFloat("Action", row, float64(ss.Action))
	if ss.Gated {
		dt.SetCellFloat("Gated", row, 1)
	} else {
		dt.SetCellFloat("Gated", row, 0)
	}
	if ss.Rewarded {
		dt.SetCellFloat("Rewarded", row, 1)
	} else {
		dt.SetCellFloat("Rewarded", row, 0)
	}
	dt.SetCellFloat("DA", row, ss.DASignal)
}

// ConfigTstTrlLog configures the testing trial log.
func (ss *Sim) ConfigTstTrlLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstTrlLog")
	dt.SetMetaData("desc", "Record of each test trial")
	dt.SetMetaData("read-only", "true")
	sch := etable.Schema{
		{"Epoch", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"Action", etensor.INT64, nil, nil},
		{"Gated", etensor.BOOL, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

// LogTstTrl logs one test trial.
func (ss *Sim) LogTstTrl(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)
	dt.SetCellFloat("Epoch", row, float64(ss.Epoch))
	dt.SetCellFloat("Trial", row, float64(ss.Trial))
	dt.SetCellString("TrialName", row, ss.TrialName)
	dt.SetCellFloat("Action", row, float64(ss.Action))
	if ss.Gated {
		dt.SetCellFloat("Gated", row, 1)
	} else {
		dt.SetCellFloat("Gated", row, 0)
	}
}

// ConfigTrnEpcLog configures the training epoch log.
func (ss *Sim) ConfigTrnEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TrnEpcLog")
	dt.SetMetaData("desc", "Record per training epoch")
	dt.SetMetaData("read-only", "true")
	sch := etable.Schema{
		{"Epoch", etensor.INT64, nil, nil},
		{"PctR1_8020", etensor.FLOAT64, nil, nil},
		{"PctR3_6040", etensor.FLOAT64, nil, nil},
		{"PctGated", etensor.FLOAT64, nil, nil},
		{"PctRewarded", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

// LogTrnEpc computes and logs training epoch summary stats.
func (ss *Sim) LogTrnEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)
	dt.SetCellFloat("Epoch", row, float64(ss.Epoch))

	trl := ss.TrnTrlLog
	nRows := trl.Rows
	startRow := nRows - ss.NTrialsPerEpoch
	if startRow < 0 {
		startRow = 0
	}

	// Count action selections for this epoch
	var n8020, nR1, n6040, nR3, nGated, nRew, nTotal int
	for r := startRow; r < nRows; r++ {
		tn := trl.CellString("TrialName", r)
		act := int(trl.CellFloat("Action", r))
		gated := trl.CellFloat("Gated", r) > 0.5
		rew := trl.CellFloat("Rewarded", r) > 0.5
		nTotal++
		if gated {
			nGated++
		}
		if rew {
			nRew++
		}
		if strings.Contains(tn, "8020") {
			n8020++
			if act == 1 && gated {
				nR1++
			}
		} else if strings.Contains(tn, "6040") {
			n6040++
			if act == 3 && gated {
				nR3++
			}
		}
	}

	pctR1 := 0.0
	if n8020 > 0 {
		pctR1 = float64(nR1) / float64(n8020)
	}
	pctR3 := 0.0
	if n6040 > 0 {
		pctR3 = float64(nR3) / float64(n6040)
	}
	pctGated := 0.0
	if nTotal > 0 {
		pctGated = float64(nGated) / float64(nTotal)
	}
	pctRew := 0.0
	if nTotal > 0 {
		pctRew = float64(nRew) / float64(nTotal)
	}

	dt.SetCellFloat("PctR1_8020", row, pctR1)
	dt.SetCellFloat("PctR3_6040", row, pctR3)
	dt.SetCellFloat("PctGated", row, pctGated)
	dt.SetCellFloat("PctRewarded", row, pctRew)

	if ss.TrnEpcPlot != nil {
		ss.TrnEpcPlot.GoUpdate()
	}
}

// ConfigTstEpcLog configures the testing epoch log.
func (ss *Sim) ConfigTstEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstEpcLog")
	dt.SetMetaData("desc", "Record per test epoch")
	dt.SetMetaData("read-only", "true")
	sch := etable.Schema{
		{"Epoch", etensor.INT64, nil, nil},
		{"ChooseA", etensor.FLOAT64, nil, nil},
		{"AvoidB", etensor.FLOAT64, nil, nil},
		{"PctGated", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

// LogTstEpc computes and logs test epoch summary statistics.
// Choose-A: select R1 over R3/R4 when both S1+S2 present
// Avoid-B: avoid R2 compared to R3/R4
// For simplicity in this version, we compute:
// Choose-A ~ fraction choosing R1 on 8020 trials
// Avoid-B ~ fraction avoiding R2 on 8020 trials (choosing anything else)
func (ss *Sim) LogTstEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	trl := ss.TstTrlLog
	nRows := trl.Rows
	startRow := nRows - ss.NTrialsPerEpoch
	if startRow < 0 {
		startRow = 0
	}

	var n8020, nR1, nNotR2, n8020Gated, nGated, nTotal int
	for r := startRow; r < nRows; r++ {
		tn := trl.CellString("TrialName", r)
		act := int(trl.CellFloat("Action", r))
		gated := trl.CellFloat("Gated", r) > 0.5
		nTotal++
		if gated {
			nGated++
		}
		if strings.Contains(tn, "8020") && gated {
			n8020++
			n8020Gated++
			if act == 1 {
				nR1++
			}
			if act != 2 {
				nNotR2++
			}
		}
	}

	chooseA := 0.0
	if n8020 > 0 {
		chooseA = float64(nR1) / float64(n8020)
	}
	avoidB := 0.0
	if n8020 > 0 {
		avoidB = float64(nNotR2) / float64(n8020)
	}
	pctGated := 0.0
	if nTotal > 0 {
		pctGated = float64(nGated) / float64(nTotal)
	}

	dt.SetCellFloat("Epoch", row, float64(ss.Epoch))
	dt.SetCellFloat("ChooseA", row, chooseA)
	dt.SetCellFloat("AvoidB", row, avoidB)
	dt.SetCellFloat("PctGated", row, pctGated)

	if ss.TstEpcPlot != nil {
		ss.TstEpcPlot.GoUpdate()
	}
}

// ConfigRunLog configures the run-level log.
func (ss *Sim) ConfigRunLog(dt *etable.Table) {
	dt.SetMetaData("name", "RunLog")
	dt.SetMetaData("desc", "Summary per run")
	dt.SetMetaData("read-only", "true")
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"ChooseA", etensor.FLOAT64, nil, nil},
		{"AvoidB", etensor.FLOAT64, nil, nil},
		{"PctGated", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

// LogRun logs summary statistics for one completed run.
func (ss *Sim) LogRun(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)
	dt.SetCellFloat("Run", row, float64(ss.Run))

	// Average the test epoch stats
	tst := ss.TstEpcLog
	if tst.Rows == 0 {
		return
	}
	ix := etable.NewIdxView(tst)
	chooseA := agg.Mean(ix, "ChooseA")[0]
	avoidB := agg.Mean(ix, "AvoidB")[0]
	pctGated := agg.Mean(ix, "PctGated")[0]
	dt.SetCellFloat("ChooseA", row, chooseA)
	dt.SetCellFloat("AvoidB", row, avoidB)
	dt.SetCellFloat("PctGated", row, pctGated)

	if ss.RunPlot != nil {
		ss.RunPlot.GoUpdate()
	}
}

////////////////////////////////////////////////////////////////////
//  GUI

// UpdateView updates the network view if available.
func (ss *Sim) UpdateView(train bool) {
	if ss.NetView != nil && ss.NetView.IsVisible() {
		ss.NetView.Record(ss.Counters(), ss.Time.CycleTot)
		ss.NetView.GoUpdate()
	}
}

// ConfigGui creates and configures the main GUI window.
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("bg_action")
	gi.SetAppAbout("BG Action Selection Model -- Probabilistic Selection Task\nTranslated from BG_4s_inhib_PS_e7a.proj (emergent 7)")

	win := gi.NewMainWindow("bg_action", "BG Action Selection", width, height)
	ss.Win = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := gi.AddNewToolBar(mfr, "ToolBar")
	tbar.SetStretchMaxWidth()
	ss.ToolBar = tbar

	split := gi.AddNewSplitView(mfr, "Split")
	split.Dim = 0 // X dimension
	split.SetStretchMaxWidth()
	split.SetStretchMaxHeight()

	sv := giv.AddNewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := gi.AddNewTabView(split, "tv")

	nv := tv.AddNewTab(netview.KiT_NetView, "NetView").(*netview.NetView)
	ss.NetView = nv
	nv.Var = "Act"
	nv.SetNet(ss.Net)

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "TrnEpcPlot").(*eplot.Plot2D)
	ss.TrnEpcPlot = plt
	plt.SetTable(ss.TrnEpcLog)
	plt.Params.Title = "Training Epoch"
	plt.Params.XAxisCol = "Epoch"
	plt.SetColParams("PctR1_8020", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PctR3_6040", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PctGated", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)

	plt2 := tv.AddNewTab(eplot.KiT_Plot2D, "TstEpcPlot").(*eplot.Plot2D)
	ss.TstEpcPlot = plt2
	plt2.SetTable(ss.TstEpcLog)
	plt2.Params.Title = "Test Epoch"
	plt2.Params.XAxisCol = "Epoch"
	plt2.SetColParams("ChooseA", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt2.SetColParams("AvoidB", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)

	plt3 := tv.AddNewTab(eplot.KiT_Plot2D, "RunPlot").(*eplot.Plot2D)
	ss.RunPlot = plt3
	plt3.SetTable(ss.RunLog)
	plt3.Params.Title = "Run Summary"
	plt3.Params.XAxisCol = "Run"
	plt3.SetColParams("ChooseA", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt3.SetColParams("AvoidB", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)

	split.SetSplits(.2, .8)

	tbar.AddAction(gi.ActOpts{Label: "Init", Icon: "update", Tooltip: "Initialize network and environment"}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Init()
		vp.SetNeedsFullRender()
	})
	tbar.AddAction(gi.ActOpts{Label: "Train", Icon: "run", Tooltip: "Run training"}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			go ss.Train()
		}
	})
	tbar.AddAction(gi.ActOpts{Label: "Test", Icon: "run", Tooltip: "Run testing"}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			go ss.Test()
		}
	})
	tbar.AddAction(gi.ActOpts{Label: "Train+Test", Icon: "run", Tooltip: "Run full train and test"}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			go ss.TrainAndTest()
		}
	})
	tbar.AddAction(gi.ActOpts{Label: "Batch", Icon: "run", Tooltip: "Run batch of networks"}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			go ss.RunBatch()
		}
	})
	tbar.AddAction(gi.ActOpts{Label: "Stop", Icon: "stop", Tooltip: "Stop running"}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Stop()
	})
	tbar.AddAction(gi.ActOpts{Label: "New Seed", Icon: "new", Tooltip: "New random seed"}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.NewRndSeed()
	})

	vp.UpdateEndNoSig(updt)

	// main menu
	appnm := gi.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})
	amen := win.MainMenu.ChildByName(appnm, 0).(*gi.Action)
	amen.Menu.AddAppMenu(win)
	emen := win.MainMenu.ChildByName("Edit", 1).(*gi.Action)
	emen.Menu.AddCopyCutPaste(win)

	win.MainMenuUpdated()
	return win
}

// SimProps defines the properties for the Sim type in the GUI.
var SimProps = ki.Props{
	"ToolBar": ki.PropSlice{},
}

////////////////////////////////////////////////////////////////////
//  Command-line args

// CmdArgs handles command-line arguments for batch mode.
func (ss *Sim) CmdArgs() {
	var nogui bool
	var nruns int
	var nepochs int

	nogui = true
	nruns = ss.NRuns
	nepochs = ss.MaxEpochs

	if len(os.Args) > 1 {
		for i := 1; i < len(os.Args); i++ {
			arg := os.Args[i]
			switch arg {
			case "-runs":
				if i+1 < len(os.Args) {
					n, err := strconv.Atoi(os.Args[i+1])
					if err == nil {
						nruns = n
					}
					i++
				}
			case "-epochs":
				if i+1 < len(os.Args) {
					n, err := strconv.Atoi(os.Args[i+1])
					if err == nil {
						nepochs = n
					}
					i++
				}
			case "-pd":
				ss.NumIntactSNc = 2
			}
		}
	}

	_ = nogui
	ss.NRuns = nruns
	ss.MaxEpochs = nepochs
	ss.Init()
	ss.RunBatch()

	// Print summary
	fmt.Printf("Batch complete: %d runs, %d epochs\n", nruns, nepochs)
	if ss.RunLog.Rows > 0 {
		rix := etable.NewIdxView(ss.RunLog)
		chooseA := agg.Mean(rix, "ChooseA")[0]
		avoidB := agg.Mean(rix, "AvoidB")[0]
		pctGated := agg.Mean(rix, "PctGated")[0]
		fmt.Printf("Mean ChooseA: %.3f  AvoidB: %.3f  PctGated: %.3f\n", chooseA, avoidB, pctGated)
	}

	// Save logs
	ss.TrnEpcLog.SaveCSV(gi.FileName("bg_action_trn_epc.csv"), ',', true)
	ss.TstEpcLog.SaveCSV(gi.FileName("bg_action_tst_epc.csv"), ',', true)
	ss.RunLog.SaveCSV(gi.FileName("bg_action_run.csv"), ',', true)
}

// Ensure all imports are used
var (
	_ = bytes.NewBuffer
	_ = fmt.Println
	_ = log.Println
	_ = math.Abs
	_ = rand.Intn
	_ = os.Exit
	_ = strconv.Atoi
	_ = strings.Contains
	_ = time.Now
	_ = split.GroupBy
)
