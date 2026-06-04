// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// State Space Model (SSM) for sequence prediction.
// CLPS1492 Computational Cognitive Neuroscience -- Midterm
// Please see the associated README.md file for a description of this project.
//
// This file implements a proper State Space Model (SSM):
//
//	x(t) = A·x(t-1) + B·u(t)     state update
//	y(t) = C·x(t)                 output (read-out via State→Hidden→Output)
//
// where:
//   - x is the State layer (n units, one layer — not two independent context layers)
//   - A is an n×n state-transition matrix stored in Sim.A
//   - B is a scalar FmHid applied element-wise from the Hidden layer (B matrix)
//   - C is implemented by the learned State→Hidden Leabra path
//
// Three options control the A matrix at run time (all togglable in the GUI):
//   - ADiagonal: only the diagonal of A is used (reduces to n independent leaky integrators).
//     When combined with AHiPPO, the HiPPO diagonal pattern is extracted and rescaled to
//     [0.1, 0.9] so the progressively-decreasing structure is preserved.
//   - AFixed:    A is frozen after initialisation (not updated during training)
//   - AHiPPO:   initialise A with the HiPPO-LegS prior (Gu et al., 2020)
//
// When AFixed is false, A is updated each trial with the XCAL learning rule (BCM Hebbian),
// the same rule used for the standard Leabra weight updates in the rest of the network.

package main

//go:generate core generate -add-types

import (
	"embed"
	"math"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/core"
	"cogentcore.org/core/enums"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
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
	"github.com/emer/emergent/v2/paths"
	"github.com/emer/etensor/tensor/table"
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

	// FmHid is the scalar B gain: how much of Hidden(t-1) drives the State update.
	// Corresponds to the B matrix in x(t) = A·x(t-1) + B·u(t).
	FmHid float32

	// ADiagonal, if true, restricts the A matrix to its diagonal only.
	// Each state unit then only receives its own previous activation (n independent leaky integrators).
	// If false the full n×n A matrix is used, allowing cross-unit mixing.
	ADiagonal bool

	// AFixed, if true, freezes the A matrix after initialisation.
	// Combine with AHiPPO to use a fixed HiPPO prior throughout training.
	AFixed bool

	// AHiPPO, if true, initialises A with the HiPPO-LegS prior (Gu et al., 2020).
	// The continuous-time generator is Euler-discretised with dt = 1/N.
	// If false, A is initialised to a scaled identity (diagonal = 0.5).
	AHiPPO bool

	// ALrnRate is the learning rate used to update A each trial when AFixed is false.
	// XCAL rule: ΔA[i][j] = ALrnRate * XCAL(LinearState[i]*APrev[j], AAvgL[i])
	// where LinearState[i] is the pre-clamp linear SSM output and AAvgL[i] is the BCM threshold.
	ALrnRate float32

	// AAvgLTau is the time constant for the long-term running average AAvgL (default 10).
	// Matches the Leabra AvgL.Tau parameter.
	AAvgLTau float32

	// AAvgLGain is the gain multiplier applied to LinearState when updating AAvgL (default 2.5).
	// Matches the Leabra AvgL.Gain parameter.
	AAvgLGain float32

	// AAvgLMin is the minimum value of AAvgL (default 0.2).
	// Matches the Leabra AvgL.Min parameter.
	AAvgLMin float32

	// A is the n×n state-transition matrix, stored row-major (A[i*n+j]).
	// It is initialised by InitA() and optionally updated by UpdateA().
	A []float32 `display:"-"`

	// APrev stores the State layer's plus-phase activations from the previous trial,
	// i.e. x(t-1), used in both the state update and the A learning rule.
	APrev []float32 `display:"-"`

	// TmpHid is scratch storage for the Hidden layer's plus-phase activations.
	TmpHid []float32 `display:"-"`

	// LinearState stores the pre-clamp linear SSM output x̃(t) = A·x(t-1) + B·u(t)
	// before it is clamped to [0,1].  Used by UpdateA for the XCAL learning rule.
	LinearState []float32 `display:"-"`

	// AAvgL is the long-term running average of LinearState, used as the BCM threshold
	// in the XCAL weight update rule (one value per state unit).
	// Updated each trial: AAvgL[i] += (1/AAvgLTau) * (AAvgLGain*LinearState[i] - AAvgL[i]).
	AAvgL []float32 `display:"-"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.FmHid = 1.0
	ss.ADiagonal = false
	ss.AFixed = false
	ss.AHiPPO = false
	ss.ALrnRate = 0.01
	ss.AAvgLTau = 10
	ss.AAvgLGain = 2.5
	ss.AAvgLMin = 0.2

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

	// State is the single SSM state vector x(t).  It is an InputLayer whose
	// activations are set manually in ApplyInputs() using the A matrix and
	// the previous hidden-layer activity (B term).
	state := net.AddLayer2D("State", 6, 5, leabra.InputLayer)

	full := paths.NewFull()

	net.ConnectLayers(inp, hid, full, leabra.ForwardPath)
	net.BidirConnectLayers(hid, out, full)
	// C matrix: State drives Hidden (read-out path)
	net.ConnectLayers(state, hid, full, leabra.ForwardPath)

	out.PlaceAbove(hid)
	state.PlaceRightOf(inp, 2)

	net.Build()
	net.Defaults()
	ss.ApplyParams()
	net.InitWeights()
	ss.InitA()
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
		// Update A matrix at the end of every trial (no-op when AFixed is true).
		stack.Loops[etime.Trial].OnEnd.Add("UpdateA", func() {
			ss.UpdateA()
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

// InitA initialises (or re-initialises) the A state-transition matrix.
// Call this after ConfigNet, and whenever AHiPPO or ADiagonal changes at run time.
func (ss *Sim) InitA() {
	n := len(ss.Net.LayerByName("State").Neurons)
	ss.A = make([]float32, n*n)
	ss.APrev = make([]float32, n)
	ss.TmpHid = make([]float32, n)
	ss.LinearState = make([]float32, n)
	ss.AAvgL = make([]float32, n)
	for i := range ss.AAvgL {
		ss.AAvgL[i] = ss.AAvgLMin // initialise at minimum threshold
	}

	if ss.AHiPPO {
		if ss.ADiagonal {
			// Diagonal-only mode with HiPPO: extract the HiPPO diagonal pattern
			// and rescale it to a useful range so the structure is preserved.
			ss.initHiPPODiagonal(n)
		} else {
			ss.initHiPPO(n)
		}
	} else {
		// Default: scaled identity — each state unit retains half its previous value.
		for i := 0; i < n; i++ {
			ss.A[i*n+i] = 0.5
		}
	}
}

// initHiPPO sets A to the discretised HiPPO-LegS matrix (Gu et al., 2020).
//
// The continuous-time generator is:
//
//	A_c[i][k] = −√((2i+1)(2k+1)) / N   for i > k   (lower triangular)
//	A_c[i][i] = −(i+1) / N              for i = k   (diagonal; i is 0-based row index)
//	A_c[i][k] = 0                        for i < k
//
// where N is the state dimension.  The forward-Euler discretisation
// A_d = I + dt·A_c with dt = 1/N is used to obtain a stable transition matrix.
func (ss *Sim) initHiPPO(N int) {
	dt := float32(1.0) / float32(N)
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			var ac float32
			if i > j {
				ac = -float32(math.Sqrt(float64((2*i+1)*(2*j+1)))) / float32(N)
			} else if i == j {
				// diagonal: A_c[i][i] = -(i+1)/N  (i is the 0-based row index)
				ac = -float32(i+1) / float32(N)
			}
			if i == j {
				ss.A[i*N+j] = 1 + dt*ac // I + dt*A_c diagonal
			} else {
				ss.A[i*N+j] = dt * ac // dt*A_c off-diagonal
			}
		}
	}
}

// initHiPPODiagonal initialises only the diagonal of A with the HiPPO-LegS pattern
// (Gu et al., 2020), rescaled so the progressively-decreasing structure spans a
// useful weight range [0.1, 0.9] instead of the narrow raw range (e.g., [0.90, 0.99]
// for N=10).  The 0-th unit gets the largest weight (most recent memory) and the
// (N-1)-th unit gets the smallest (most remote memory).
func (ss *Sim) initHiPPODiagonal(N int) {
	// Raw HiPPO diagonal after forward-Euler: A_d[i][i] = 1 - (i+1)/N²
	// This decreases from v[0] = 1 - 1/N² (largest) to v[N-1] = 1 - 1/N (smallest).
	vMax := float32(1) - float32(1)/float32(N*N)
	vMin := float32(1) - float32(1)/float32(N)
	span := vMax - vMin
	const lo, hi = float32(0.1), float32(0.9)
	for i := 0; i < N; i++ {
		v := float32(1) - float32(i+1)/float32(N*N)
		if span > 1e-6 {
			ss.A[i*N+i] = lo + (v-vMin)/span*(hi-lo)
		} else {
			ss.A[i*N+i] = (lo + hi) / 2
		}
	}
}

// xcalDWt implements the XCAL "check mark" weight change function used throughout
// the Leabra network.  srval is the sender×receiver co-product; thrP is the BCM
// threshold (AAvgL for the A-matrix learning rule).
// Matches XCalParams.DWt with default parameters DRev=0.1, DThr=0.0001.
func xcalDWt(srval, thrP float32) float32 {
	const (
		dRev      = 0.1
		dThr      = 0.0001
		dRevRatio = -(1 - dRev) / dRev // = -9.0
	)
	if srval < dThr {
		return 0
	} else if srval > thrP*dRev {
		return srval - thrP
	}
	return srval * dRevRatio
}

// UpdateA updates the A matrix using the XCAL learning rule (BCM Hebbian),
// the same rule used for standard Leabra weight updates in the rest of the network:
//
//	ΔA[i][j] = ALrnRate · XCAL(y[i]·prev_state[j], AAvgL[i])
//
// where y[i] is the pre-clamp linear SSM output (LinearState[i]) and AAvgL[i]
// is the BCM floating threshold — a gain-scaled exponential moving average of y[i].
// AAvgL is updated each trial before the weight change:
//
//	AAvgL[i] += (1/AAvgLTau) · (AAvgLGain·y[i] − AAvgL[i])
//
// The XCAL function naturally prevents weight saturation: when the BCM threshold
// rises (because a unit is consistently highly active) it suppresses further LTP
// and promotes LTD, keeping weights in a healthy range.
// UpdateA is a no-op when AFixed is true.
func (ss *Sim) UpdateA() {
	if ss.AFixed {
		return
	}

	n := len(ss.APrev)
	if n == 0 || len(ss.A) != n*n || len(ss.LinearState) != n {
		return // not yet populated
	}

	// Ensure AAvgL is initialised (guard against first-trial edge case).
	if len(ss.AAvgL) != n {
		ss.AAvgL = make([]float32, n)
		for i := range ss.AAvgL {
			ss.AAvgL[i] = ss.AAvgLMin
		}
	}

	// Update long-term running average (BCM threshold), matching Leabra AvgL update.
	dt := float32(1.0) / ss.AAvgLTau
	for i := 0; i < n; i++ {
		ss.AAvgL[i] += dt * (ss.AAvgLGain*ss.LinearState[i] - ss.AAvgL[i])
		if ss.AAvgL[i] < ss.AAvgLMin {
			ss.AAvgL[i] = ss.AAvgLMin
		}
	}

	lr := ss.ALrnRate
	useDiag := ss.ADiagonal
	if useDiag {
		for i := 0; i < n; i++ {
			srval := ss.LinearState[i] * ss.APrev[i]
			ss.A[i*n+i] += lr * xcalDWt(srval, ss.AAvgL[i])
		}
	} else {
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				srval := ss.LinearState[i] * ss.APrev[j]
				ss.A[i*n+j] += lr * xcalDWt(srval, ss.AAvgL[i])
			}
		}
	}
}

// ApplyInputs applies input patterns from the environment and updates the SSM
// State layer.  The state update implements the SSM equations:
//
//	x(t) = A·x(t-1) + FmHid·hidden(t-1)
//
// where A is the n×n matrix stored in Sim.A (optionally restricted to its
// diagonal when ADiagonal is true), and FmHid is the scalar B gain.
func (ss *Sim) ApplyInputs() {

	ctx := &ss.Context
	net := ss.Net

	ev := ss.Envs.ByMode(ctx.Mode).(*env.FixedTable)
	ev.Step()

	net.InitExt()

	lays := net.LayersByType(leabra.InputLayer, leabra.TargetLayer)

	ss.Stats.SetString("TrialName", ev.TrialName.Cur)
	for _, lnm := range lays {
		// Skip the State layer: it is InputLayer type but is set manually below.
		if lnm == "State" {
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

	// Collect Hidden(t-1) plus-phase activations for the B term.
	hid := net.LayerByName("Hidden")
	hid.UnitValues(&ss.TmpHid, "ActP", 0)

	// Collect State(t-1) plus-phase activations for the A term.
	stateLay := net.LayerByName("State")
	stateLay.UnitValues(&ss.APrev, "ActP", 0)

	n := len(ss.APrev)
	// Guard: if A matrix not yet initialised (e.g. very first trial), initialise now.
	if len(ss.A) != n*n {
		ss.InitA()
	}
	newState := make([]float32, n)

	// x(t) = A·x(t-1) + FmHid·hidden(t-1)
	// When ADiagonal is true, only the diagonal of A is used regardless of AHiPPO
	// (the HiPPO diagonal structure is preserved in InitA via initHiPPODiagonal).
	useDiag := ss.ADiagonal
	for i := 0; i < n; i++ {
		var aterm float32
		if useDiag {
			// Only the diagonal: x[i](t) = A[i][i]*x[i](t-1)
			aterm = ss.A[i*n+i] * ss.APrev[i]
		} else {
			// Full A matrix: x[i](t) = sum_j A[i][j]*x[j](t-1)
			for j := 0; j < n; j++ {
				aterm += ss.A[i*n+j] * ss.APrev[j]
			}
		}
		v := aterm + ss.FmHid*ss.TmpHid[i]
		// Save the pre-clamp linear output for the XCAL learning rule.
		ss.LinearState[i] = v
		// Clamp to [0,1] so the InputLayer activation stays in a sensible range.
		if v < 0 {
			v = 0
		} else if v > 1 {
			v = 1
		}
		newState[i] = v
	}

	// Drive the State layer with the computed activations.
	clr, set, toTarg := stateLay.ApplyExtFlags()
	for i := range stateLay.Neurons {
		stateLay.ApplyExtValue(i, newState[i], clr, set, toTarg)
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
	ss.InitA() // re-initialise A matrix for this run
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
	ss.GUI.MakeBody(ss, "SSM", title, `SSM midterm -- proper state space model with a single n×n A matrix`)
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
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{
		Label:   "Re-init A",
		Icon:    icons.Reset,
		Tooltip: "Re-initialise the A state-transition matrix using the current ADiagonal / AHiPPO settings. Useful after changing those flags mid-run.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.InitA()
			ss.ViewUpdate.Update()
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
