package evaluator

import (
	"os"
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBuildGraphMLP(t *testing.T) {
	data, err := os.ReadFile("testdata/mlp.orion")
	require.NoError(t, err)

	header, _, err := ParseContainer(data)
	require.NoError(t, err)

	g, err := buildGraph(header)
	require.NoError(t, err)

	// Verify basic graph properties.
	assert.Equal(t, "flatten", g.Input)
	assert.Equal(t, "fc2", g.Output)
	assert.Equal(t, 4, len(g.Nodes))
	assert.Equal(t, 4, len(g.Order))

	// Verify all nodes are in the order.
	orderSet := make(map[string]bool)
	for _, name := range g.Order {
		orderSet[name] = true
	}
	assert.True(t, orderSet["flatten"])
	assert.True(t, orderSet["fc1"])
	assert.True(t, orderSet["act1"])
	assert.True(t, orderSet["fc2"])

	// Verify topological ordering: flatten before fc1 before act1 before fc2.
	indexOf := make(map[string]int)
	for i, name := range g.Order {
		indexOf[name] = i
	}
	assert.Less(t, indexOf["flatten"], indexOf["fc1"])
	assert.Less(t, indexOf["fc1"], indexOf["act1"])
	assert.Less(t, indexOf["act1"], indexOf["fc2"])
}

func TestBuildGraphCyclicEdges(t *testing.T) {
	header := &CompiledHeader{
		Graph: HeaderGraph{
			Input:  "a",
			Output: "c",
			Nodes: []HeaderNode{
				{Name: "a", Op: "flatten"},
				{Name: "b", Op: "quad"},
				{Name: "c", Op: "quad"},
			},
			Edges: []HeaderEdge{
				{Src: "a", Dst: "b"},
				{Src: "b", Dst: "c"},
				{Src: "c", Dst: "b"}, // cycle: c -> b
			},
		},
	}

	_, err := buildGraph(header)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "cycle detected")
}

func TestBuildGraphNonexistentNode(t *testing.T) {
	header := &CompiledHeader{
		Graph: HeaderGraph{
			Input:  "a",
			Output: "b",
			Nodes: []HeaderNode{
				{Name: "a", Op: "flatten"},
				{Name: "b", Op: "quad"},
			},
			Edges: []HeaderEdge{
				{Src: "a", Dst: "ghost"}, // ghost doesn't exist
			},
		},
	}

	_, err := buildGraph(header)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "nonexistent destination node")
}

func TestBuildGraphNonexistentSrcNode(t *testing.T) {
	header := &CompiledHeader{
		Graph: HeaderGraph{
			Input:  "a",
			Output: "b",
			Nodes: []HeaderNode{
				{Name: "a", Op: "flatten"},
				{Name: "b", Op: "quad"},
			},
			Edges: []HeaderEdge{
				{Src: "ghost", Dst: "b"}, // ghost doesn't exist
			},
		},
	}

	_, err := buildGraph(header)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "nonexistent source node")
}

func TestBuildGraphInputsMap(t *testing.T) {
	data, err := os.ReadFile("testdata/mlp.orion")
	require.NoError(t, err)

	header, _, err := ParseContainer(data)
	require.NoError(t, err)

	g, err := buildGraph(header)
	require.NoError(t, err)

	// flatten has no predecessors.
	assert.Empty(t, g.Inputs["flatten"])

	// fc1 has exactly one predecessor: flatten.
	assert.Equal(t, []string{"flatten"}, g.Inputs["fc1"])

	// act1 has exactly one predecessor: fc1.
	assert.Equal(t, []string{"fc1"}, g.Inputs["act1"])

	// fc2 has exactly one predecessor: act1.
	assert.Equal(t, []string{"act1"}, g.Inputs["fc2"])
}

func TestBuildGraphNodeProperties(t *testing.T) {
	data, err := os.ReadFile("testdata/mlp.orion")
	require.NoError(t, err)

	header, _, err := ParseContainer(data)
	require.NoError(t, err)

	g, err := buildGraph(header)
	require.NoError(t, err)

	// Verify node ops are carried through.
	assert.Equal(t, "flatten", g.Nodes["flatten"].Op)
	assert.Equal(t, "linear_transform", g.Nodes["fc1"].Op)
	assert.Equal(t, "quad", g.Nodes["act1"].Op)
	assert.Equal(t, "linear_transform", g.Nodes["fc2"].Op)

	// Verify fc1 has blob refs.
	assert.NotEmpty(t, g.Nodes["fc1"].BlobRefs)
}

func TestBuildGraphDuplicateNode(t *testing.T) {
	header := &CompiledHeader{
		Graph: HeaderGraph{
			Input:  "a",
			Output: "a",
			Nodes: []HeaderNode{
				{Name: "a", Op: "flatten"},
				{Name: "a", Op: "quad"}, // duplicate
			},
			Edges: []HeaderEdge{},
		},
	}

	_, err := buildGraph(header)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "duplicate node name")
}

func TestBuildGraphMissingInputNode(t *testing.T) {
	header := &CompiledHeader{
		Graph: HeaderGraph{
			Input:  "missing",
			Output: "a",
			Nodes: []HeaderNode{
				{Name: "a", Op: "flatten"},
			},
			Edges: []HeaderEdge{},
		},
	}

	_, err := buildGraph(header)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "input node")
}

func TestBuildGraphMissingOutputNode(t *testing.T) {
	header := &CompiledHeader{
		Graph: HeaderGraph{
			Input:  "a",
			Output: "missing",
			Nodes: []HeaderNode{
				{Name: "a", Op: "flatten"},
			},
			Edges: []HeaderEdge{},
		},
	}

	_, err := buildGraph(header)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "output node")
}

func TestBuildGraphSigmoid(t *testing.T) {
	data, err := os.ReadFile("testdata/sigmoid.orion")
	require.NoError(t, err)

	header, _, err := ParseContainer(data)
	require.NoError(t, err)

	g, err := buildGraph(header)
	require.NoError(t, err)

	assert.Equal(t, 4, len(g.Order))

	// Verify polynomial node is present.
	assert.Equal(t, "polynomial", g.Nodes["act1"].Op)

	// Verify topological order.
	indexOf := make(map[string]int)
	for i, name := range g.Order {
		indexOf[name] = i
	}
	// All predecessors come before their successors.
	for name, preds := range g.Inputs {
		for _, pred := range preds {
			assert.Less(t, indexOf[pred], indexOf[name],
				"%s should come before %s in topological order", pred, name)
		}
	}

	// Verify order is deterministic by checking all nodes covered.
	ordered := make([]string, len(g.Order))
	copy(ordered, g.Order)
	sort.Strings(ordered)
	expected := []string{"act1", "fc1", "fc2", "flatten"}
	assert.Equal(t, expected, ordered)
}
