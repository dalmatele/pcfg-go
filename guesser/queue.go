package guesser

import (
	"container/heap"
	"runtime"
	"sync"
	"sync/atomic"
	"fmt"
	"os"

	pcfg "github.com/cyclone-github/pcfg-go/shared"
)

// wrap PTItem for the priority queue
type queueItem struct {
	item  pcfg.PTItem
	seq   int64 // insertion order for stable tie-breaking
	index int   // heap index
}

// priorityQueue implements heap.Interface for max-probability ordering
type priorityQueue []*queueItem

func (pq priorityQueue) Len() int { return len(pq) }

func (pq priorityQueue) Less(i, j int) bool {
	if pq[i].item.Prob != pq[j].item.Prob {
		return pq[i].item.Prob > pq[j].item.Prob
	}
	// Stable tie-breaking: earlier insertions come first
	return pq[i].seq < pq[j].seq
}

func (pq priorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *priorityQueue) Push(x interface{}) {
	n := len(*pq)
	item := x.(*queueItem)
	item.index = n
	*pq = append(*pq, item)
}

func (pq *priorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	item.index = -1
	*pq = old[0 : n-1]
	return item
}

// manage priority queue for guess generation
type PcfgQueue struct {
	pq             priorityQueue
	grammar        pcfg.Grammar
	base           []pcfg.BaseStructure
	MaxProbability float64
	MinProbability float64
	seqCounter     atomic.Int64

	// Các biến cho Fast Restore
    treesToSkip     int64   // Số cây cần skip
    initialSkip     int64   // Lưu lại giá trị ban đầu để log
    isRestoring     bool    // Đang ở chế độ restore
}

// create and initializes a priority queue with base structures
func NewPcfgQueue(grammar pcfg.Grammar, base []pcfg.BaseStructure) *PcfgQueue {
	return newPcfgQueueWithSave(grammar, base, 0, 1)
}

// creates queue restored from session (minProb, maxProb)
func NewPcfgQueueFromSave(grammar pcfg.Grammar, base []pcfg.BaseStructure, minProb, maxProb float64) *PcfgQueue {
	return newPcfgQueueWithSave(grammar, base, minProb, maxProb)
}

func newPcfgQueueWithSave(grammar pcfg.Grammar, base []pcfg.BaseStructure, minProb, maxProb float64) *PcfgQueue {
	q := &PcfgQueue{
		grammar:        grammar,
		base:           base,
		MaxProbability: maxProb,
		MinProbability: minProb,
	}

	heap.Init(&q.pq)

	if minProb > 0 || maxProb < 1 {
		// Restore: run tree traversal in parallel across base structures
		restoreProbOrderParallel(q, grammar, base, minProb, maxProb)
	} else {
		for _, b := range base {
			pt := make([]pcfg.PTNode, len(b.Replacements))
			for i, r := range b.Replacements {
				pt[i] = pcfg.PTNode{Type: r, Index: 0}
			}
			prob := findProb(grammar, pt, b.Prob)
			item := pcfg.PTItem{
				Prob:     prob,
				PT:       pt,
				BaseProb: b.Prob,
			}
			seq := q.seqCounter.Add(1)
			heap.Push(&q.pq, &queueItem{item: item, seq: seq})
		}
	}

	return q
}

// restoreProbOrderParallel runs restore per base structure in parallel, then merges.
func restoreProbOrderParallel(q *PcfgQueue, grammar pcfg.Grammar, base []pcfg.BaseStructure, minProb, maxProb float64) {
	numWorkers := runtime.NumCPU()
	if numWorkers < 1 {
		numWorkers = 1
	}
	if numWorkers > len(base) {
		numWorkers = len(base)
	}

	type work struct {
		b pcfg.BaseStructure
	}
	workCh := make(chan work, len(base))
	for _, b := range base {
		workCh <- work{b: b}
	}
	close(workCh)

	var mu sync.Mutex
	var allItems []pcfg.PTItem

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			local := make([]pcfg.PTItem, 0, 4096)
			for w := range workCh {
				pt := make([]pcfg.PTNode, len(w.b.Replacements))
				for i, r := range w.b.Replacements {
					pt[i] = pcfg.PTNode{Type: r, Index: 0}
				}
				prob := findProb(grammar, pt, w.b.Prob)
				item := pcfg.PTItem{
					Prob:     prob,
					PT:       pt,
					BaseProb: w.b.Prob,
				}
				restoreProbOrderToSlice(grammar, &item, minProb, maxProb, func(it *pcfg.PTItem) {
					local = append(local, *it)
				})
			}
			if len(local) > 0 {
				mu.Lock()
				allItems = append(allItems, local...)
				mu.Unlock()
			}
		}()
	}
	wg.Wait()

	for i := range allItems {
		it := &allItems[i]
		seq := q.seqCounter.Add(1)
		heap.Push(&q.pq, &queueItem{item: *it, seq: seq})
	}
}

// restoreProbOrderToSlice collects items in [minProb, maxProb] via callback (no shared state).
func restoreProbOrderToSlice(grammar pcfg.Grammar, ptItem *pcfg.PTItem, minProb, maxProb float64, collect func(*pcfg.PTItem)) {
	prob := ptItem.Prob
	if prob < minProb {
		return
	}
	if prob <= maxProb {
		if !isParentInQueue(grammar, ptItem, maxProb) {
			collect(ptItem)
		}
		return
	}
	children := findChildren(grammar, ptItem)
	for i := range children {
		restoreProbOrderToSlice(grammar, &children[i], minProb, maxProb, collect)
	}
}

// returns true if any "parent" of ptItem would be in the queue
func isParentInQueue(grammar pcfg.Grammar, ptItem *pcfg.PTItem, maxProb float64) bool {
	for pos, node := range ptItem.PT {
		if node.Index == 0 {
			continue
		}
		parent := make([]pcfg.PTNode, len(ptItem.PT))
		copy(parent, ptItem.PT)
		parent[pos] = pcfg.PTNode{Type: parent[pos].Type, Index: parent[pos].Index - 1}
		parentProb := findProb(grammar, parent, ptItem.BaseProb)
		if parentProb <= maxProb {
			return true
		}
	}
	return false
}



// returns current queue size
func (q *PcfgQueue) QueueSize() int {
	return q.pq.Len()
}

func findProb(grammar pcfg.Grammar, pt []pcfg.PTNode, baseProb float64) float64 {
	prob := baseProb
	for _, node := range pt {
		entries := grammar[node.Type]
		if node.Index < len(entries) {
			prob *= entries[node.Index].Prob
		}
	}
	return prob
}

func findChildren(grammar pcfg.Grammar, ptItem *pcfg.PTItem) []pcfg.PTItem {
	parentPT := ptItem.PT
	var children []pcfg.PTItem

	for pos, node := range parentPT {
		entries := grammar[node.Type]
		if len(entries) == node.Index+1 {
			continue
		}

		child := make([]pcfg.PTNode, len(parentPT))
		copy(child, parentPT)
		child[pos] = pcfg.PTNode{Type: child[pos].Type, Index: child[pos].Index + 1}

		if areYouMyChild(grammar, child, ptItem.BaseProb, pos, ptItem.Prob) {
			childProb := findProb(grammar, child, ptItem.BaseProb)
			children = append(children, pcfg.PTItem{
				PT:       child,
				BaseProb: ptItem.BaseProb,
				Prob:     childProb,
			})
		}
	}

	return children
}

func areYouMyChild(grammar pcfg.Grammar, child []pcfg.PTNode, baseProb float64, parentPos int, parentProb float64) bool {
	for pos, node := range child {
		if pos == parentPos {
			continue
		}
		if node.Index == 0 {
			continue
		}

		newParent := make([]pcfg.PTNode, len(child))
		copy(newParent, child)
		newParent[pos] = pcfg.PTNode{Type: newParent[pos].Type, Index: newParent[pos].Index - 1}

		newParentProb := findProb(grammar, newParent, baseProb)
		if newParentProb < parentProb {
			return false
		}
		if newParentProb == parentProb && pos < parentPos {
			return false
		}
	}
	return true
}

func NewPcfgQueueFromParseTrees(grammar pcfg.Grammar, base []pcfg.BaseStructure, numParseTrees int64) *PcfgQueue {
	fmt.Fprintf(os.Stderr, "[Fast Restore] ========== START ==========\n")
	fmt.Fprintf(os.Stderr, "[Fast Restore] numParseTrees: %d\n", numParseTrees)
	
	q := &PcfgQueue{
		grammar:     grammar,
		base:        base,
		treesToSkip: numParseTrees,
		initialSkip: numParseTrees,
		isRestoring: true,
	}
	
	heap.Init(&q.pq)
	
	fmt.Fprintf(os.Stderr, "[Fast Restore] Creating queue from %d base structures\n", len(base))
	
	// Khởi tạo queue từ base structures
	for _, b := range base {
		pt := make([]pcfg.PTNode, len(b.Replacements))
		for j, r := range b.Replacements {
			pt[j] = pcfg.PTNode{Type: r, Index: 0}
		}
		prob := findProb(grammar, pt, b.Prob)
		item := pcfg.PTItem{
			Prob:     prob,
			PT:       pt,
			BaseProb: b.Prob,
		}
		seq := q.seqCounter.Add(1)
		heap.Push(&q.pq, &queueItem{item: item, seq: seq})
	}
	
	fmt.Fprintf(os.Stderr, "[Fast Restore] Initial queue size: %d\n", q.pq.Len())
	fmt.Fprintf(os.Stderr, "[Fast Restore] treesToSkip: %d\n", q.treesToSkip)
	fmt.Fprintf(os.Stderr, "[Fast Restore] isRestoring: %v\n", q.isRestoring)
	
	// Nếu numParseTrees > queue size, chỉ skip hết queue
	if numParseTrees > int64(q.pq.Len()) && q.pq.Len() > 0 {
		fmt.Fprintf(os.Stderr, "[Fast Restore] WARNING: numParseTrees (%d) > queue size (%d)\n", 
			numParseTrees, q.pq.Len())
		fmt.Fprintf(os.Stderr, "[Fast Restore] Will skip all %d items\n", q.pq.Len())
		q.treesToSkip = int64(q.pq.Len())
	}
	
	// Nếu queue rỗng từ đầu
	if q.pq.Len() == 0 {
		fmt.Fprintf(os.Stderr, "[Fast Restore] ERROR: Queue is empty from start!\n")
		return q
	}
	
	fmt.Fprintf(os.Stderr, "[Fast Restore] Starting skip loop...\n")
	skipped := int64(0)
	
	// Skip và push children (giống như Next())
	for q.isRestoring && q.treesToSkip > 0 && q.pq.Len() > 0 {
		// Pop item
		qi := heap.Pop(&q.pq).(*queueItem)
		q.treesToSkip--
		skipped++
		
		// ✅ QUAN TRỌNG: Push children (giống như Next())
		children := findChildren(q.grammar, &qi.item)
		for _, child := range children {
			seq := q.seqCounter.Add(1)
			heap.Push(&q.pq, &queueItem{item: child, seq: seq})
		}
		
		// Log mỗi 1000 items
		if skipped%1000 == 0 {
			fmt.Fprintf(os.Stderr, "[Fast Restore] Skipped %d items, remaining: %d, queue size: %d\n", 
				skipped, q.treesToSkip, q.pq.Len())
		}
		
		// Nếu skip hết hoặc queue rỗng
		if q.treesToSkip == 0 || q.pq.Len() == 0 {
			fmt.Fprintf(os.Stderr, "[Fast Restore] Skip loop ended: skipped=%d, remaining=%d, queue_size=%d\n", 
				skipped, q.treesToSkip, q.pq.Len())
			q.isRestoring = false
			break
		}
	}
	
	fmt.Fprintf(os.Stderr, "[Fast Restore] Final queue size: %d\n", q.pq.Len())
	fmt.Fprintf(os.Stderr, "[Fast Restore] ========== DONE ==========\n")
	
	return q
}

func (q *PcfgQueue) Next() *pcfg.PTItem {
	if q.pq.Len() == 0 {
		return nil
	}
	
	// Pop item
	qi := heap.Pop(&q.pq).(*queueItem)
	
	// Cập nhật max probability
	q.MaxProbability = qi.item.Prob
	
	// Push children
	children := findChildren(q.grammar, &qi.item)
	for _, child := range children {
		seq := q.seqCounter.Add(1)
		heap.Push(&q.pq, &queueItem{item: child, seq: seq})
	}
	
	return &qi.item
}
