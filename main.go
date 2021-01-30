package main

import (
	"encoding/json"
	"fmt"
	//uuid "github.com/satori/go.uuid"
	//"github.com/sirupsen/logrus"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
)

func F() {
	k := 0
	defer func() {
		fmt.Println(k)
	}()

	k = 20
}


func getRepoMethodNameFromDefer() string {
	pc := make([]uintptr, 15)
	n := runtime.Callers(1, pc)
	frames := runtime.CallersFrames(pc[:n])
	frame, _ := frames.Next()

	return frame.Function
}

func Test() {
	var err error

	err = fmt.Errorf("Error one")
	defer func() {
		metric(err)
	}()

	err = fmt.Errorf("Error two")
	defer func() {
		metric(err)
	}()

	metric(err)
}

func metric(err error) {
	fmt.Println(getRepoMethodNameFromDefer())

	fmt.Println(err)
}

var matchFirstCap = regexp.MustCompile("([^A-Z])([A-Z][a-z]+)")
var matchAllCap   = regexp.MustCompile("([a-z0-9])([A-Z])")

func ToSnakeCase(str string) string {
	snake := matchFirstCap.ReplaceAllString(str, "${1}_${2}")
	snake  = matchAllCap.ReplaceAllString(snake, "${1}_${2}")
	return strings.ToLower(snake)
}

func Err() (string, error) {
	return "str", fmt.Errorf("Errrrrrr")
}

// About
var (
	GitCommit string
	GitBranch string
	BuildDate string
	Version   string
)

var reFirstCap   = regexp.MustCompile("([^A-Z])([A-Z][a-z]+)")
var reAllCap     = regexp.MustCompile("([a-z0-9])([A-Z])")

// toSnakeCase string camel case to snake case
func toSnakeCase(str string) string {
	snake := reFirstCap.ReplaceAllString(str, "${1}_${2}")
	snake  = reAllCap.ReplaceAllString(snake, "${1}_${2}")
	return strings.ToLower(snake)
}

func randString() string {
	letters := []string{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"}
	length := len(letters)
	return fmt.Sprintf("%s%s%s%s%s%s%s%s%s%s",
		letters[rand.Intn(length)],
		letters[rand.Intn(length)],
		letters[rand.Intn(length)],
		letters[rand.Intn(length)],
		letters[rand.Intn(length)],
		letters[rand.Intn(length)],
		letters[rand.Intn(length)],
		letters[rand.Intn(length)],
		letters[rand.Intn(length)],
		letters[rand.Intn(length)])
}

func hiHandler(w http.ResponseWriter, r *http.Request) {
	w.Write([]byte("hi"))
}

// ObjToMap преобразование объекта в map
func ObjToMap(obj interface{}) (f map[string]interface{}) {
	f = make(map[string]interface{})

	data, err := json.Marshal(obj)
	if err != nil {
		return
	}

	err = json.Unmarshal(data, &f)
	if err != nil {
		return
	}

	return
}

type D struct {
	I int32
}

type D2 struct {
	I int32 `json:"ff,omitempty"`
	I2 int32 `json:"ffff,omitempty"`
	List []string `json:"list,omitempty"`
}

func (d *D) GetI() int32 {
	return d.I
}

func (d *D2) GetI() int32 {
	return d.I
}

type ID interface {
	GetI() int32
}

func t(b []byte) {
	fmt.Println(len(b))
	b[3] = 8
}

func dd(f *D2) {
	f.List = append(f.List, "ff")
}

func isUnique(arr []string) bool {
	m := make(map[string]struct{}, len(arr))
	for _, a := range arr {
		m[a] = struct{}{}
	}

	return len(m) == len(arr)
}

func correctBrackets(s string) bool {
	arr := strings.Split(s, "")
	b := []string{}
	for _, a := range arr {
		switch a {
		case "(", "{", "[":
			b = append(b, a)
		}

		switch a {
		case ")":
			if len(b) == 0 {
				return false
			}
			if b[len(b)-1] != "(" {
				return false
			} else {
				b = b[:len(b) - 1]
			}
		case "}":
			if len(b) == 0 {
				return false
			}
			if b[len(b)-1] != "{" {
				return false
			} else {
				b = b[:len(b) - 1]
			}
		case "]":
			if len(b) == 0 {
				return false
			}
			if b[len(b)-1] != "[" {
				return false
			} else {
				b = b[:len(b) - 1]
			}
		}
	}

	return len(b) == 0
}

func twoSum(nums []int, target int) []int {
	for i := 0; i < len(nums) - 1; i++ {
		for j := i + 1; j < len(nums); j++ {
			if nums[i] + nums[j] == target {
				return []int{i, j}
			}
		}
	}

	return nil
}

type ListNode struct {
	D int
	Next *ListNode
}

func arr2ListNode(arr []int) *ListNode {
	root := &ListNode{}
	var next *ListNode
	for i := 0; i < len(arr); i++ {
		if i == 0 {
			root.D = arr[0]
			if len(arr) > 1 {
				root.Next = &ListNode{}
				next = root.Next
			}
		} else {
			next.D = arr[i]
			if i < len(arr) - 1 {
				next.Next = &ListNode{}
				next = next.Next
			}
		}

	}

	return root
}

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	arr := []int{}
	rest := false
	for {
		restNum := 0
		if l1 != nil || l2 != nil {
			l1Num, l2Num := 0, 0
			if l1 != nil {
				l1Num = l1.D
				l1 = l1.Next
			}
			if l2 != nil {
				l2Num = l2.D
				l2 = l2.Next
			}
			if rest {
				restNum = 1
				rest = false
			}
			sum := l1Num + l2Num + restNum
			arr = append(arr, sum % 10)
			if sum >= 10 {
				rest = true
			}
		} else {
			if rest {
				arr = append(arr, 1)
			}
			break
		}
	}

	return arr2ListNode(arr)
}

func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	merg := []int{}
	i, j := 0, 0
	for {
		if i == len(nums1) && j == len(nums2) {
			break
		} else if i == len(nums1) {
			merg = append(merg, nums2[j:]...)
			break
		} else if j == len(nums2) {
			merg = append(merg, nums1[i:]...)
			break
		} else {
			if nums1[i] <= nums2[j] {
				merg = append(merg, nums1[i])
				i++
			} else {
				merg = append(merg, nums2[j])
				j++
			}
		}
	}

	if len(merg) % 2 != 0 {
		f := float64(len(merg)) / 2
		return float64(merg[int(math.Floor(f))])
	}

	fmt.Println(merg, "--", len(merg) / 2, len(merg))


	return float64(merg[len(merg) / 2] + merg[len(merg) / 2 - 1]) / 2
}

func longestPalindrome(s string) string {
	arr := strings.Split(s, "")
	if isPalindrome(arr) {
		return strings.Join(arr, "")
	}
	l := len(arr)
	maxLen := l - 1
	for {
		for i := 0; i + maxLen < l; i++ {
			if isPalindrome(arr[i:i+maxLen]) {
				return strings.Join(arr[i:i+maxLen], "")
			}
		}
		maxLen--
	}

	return ""
}

func isPalindrome(arr []string) bool {
	i, j := 0, len(arr) - 1
	for {
		if i > j {
			break
		}
		if arr[i] != arr[j] {
			return false
		}
		i++
		j--
	}

	return true
}

func reverse(x int) int {
	rev := 0

	for x != 0 {
		pop := x % 10
		x /= 10
		fmt.Println(x, pop)

		temp := rev * 10 + pop
		rev = temp
	}

	return rev
}

func binaryToDigit(s string) int {
	arr := strings.Split(s, "")
	reverseArrayString(arr)
	var power, res float64
	for _, a := range arr {
		i, _ := strconv.Atoi(a)
		res += float64(i) * math.Pow(2, power)
		power++
	}

	return int(res)
}

func decimalToBinary(i int) string {
	a := []string{}
	for i > 1 {
		a = append(a, strconv.Itoa(i % 2))
		i /= 2
	}
	a = append(a, strconv.Itoa(i))

	reverseArrayString(a)

	return strings.Join(a, "")
}

func reverseArrayString(a []string) {
	from, to := 0, len(a) - 1
	for from < to {
		a[from], a[to] = a[to], a[from]
		from++
		to--
	}
}

func reverseArrayInt(a []int) {
	from, to := 0, len(a) - 1
	for from < to {
		a[from], a[to] = a[to], a[from]
		from++
		to--
	}
}

func longestSequenseOne(arr []int) int {
	maxOneCount := 0
	for _, a := range arr {
		str := decimalToBinary(a)
		fmt.Println(str)
		tmpCount := 0
		for _, s := range strings.Split(str, "") {
			if s == "1" {
				tmpCount++
				if tmpCount > maxOneCount {
					maxOneCount = tmpCount
				}
			} else {
				tmpCount = 0
			}
		}
	}

	return maxOneCount
}

func uniqFileInt(fileName string) []int {
	file, _ := os.Open(fileName)
	defer file.Close()

	lineNum := 0
	var res []int
	rest := ""
	for {
		b := make([]byte, 3)
		n, err := file.Read(b)
		str := ""
		if n > 0 {
			str = fmt.Sprintf("%s%s", rest, b)
		} else {
			str = rest
		}
		lines := strings.Split(str, "\n")
		fmt.Println(lines)
		for idx, line := range lines {
			if idx == len(lines) - 1 {
				rest = line
				if err != nil {
					if err == io.EOF {
						i, _ := strconv.Atoi(line)
						if i > res[len(res) - 1] {
							res = append(res, i)
						}
					}
				}
			} else {
				fmt.Println(line, "------", len(line))
				if lineNum == 0 {

				} else {
					i, _ := strconv.Atoi(line)
					if len(res) == 0 {
						res = append(res, i)
					} else if i > res[len(res) - 1] {
						res = append(res, i)
					}
				}

				lineNum++
			}
		}
		if err != nil {
			if err == io.EOF {
				break
			}
		}

	}

	return res
}

func genBrakets(cnt int) []string {
	var res [][]string
	res = append(res, []string{})
	for i := 0; i < cnt; i++ {
		res[0] = append(res[0], "(")
	}
	for i := 0; i < cnt; i++ {
		res[0] = append(res[0], ")")
	}

	for i := 1; i < cnt; i++ {
		for j := cnt; j < cnt * 2 - 1; j++ {
			new := make([]string, cnt * 2)
			copy(new, res[0])
			new[i], new[j] = new[j], new[i]
			res = append(res, new)
		}
	}

	var result []string
	for _, r := range res {
		result = append(result, strings.Join(r, ""))
	}

	sort.Strings(result)

	return result
}

func fact(i int) int {
	if i == 1 {
		return 1
	}
	return i * fact(i - 1)

}

func isAnagram(s1, s2 string) bool {
	if len(s1) != len(s2) {
		return false
	}

	a1 := strings.Split(s1, "")
	a2 := strings.Split(s2, "")

	sort.Strings(a1)
	sort.Strings(a2)

	s1 = strings.Join(a1, "")
	s2 = strings.Join(a2, "")

	return s1 == s2
}

func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	if l1 == nil && l2 == nil {
		return nil
	} else if l1 == nil {
		return l2
	} else if l2 == nil {
		return l1
	}

	var root *ListNode
	var next *ListNode
	if l1.D <= l2.D {
		root = l1
		l1 = l1.Next
	} else {
		root = l2
		l2 = l2.Next
	}
	next = root
	for {
		if l1 == nil && l2 == nil {
			break
		} else if l1 == nil {
			next.Next = l2
			l2 = l2.Next
		} else if l2 == nil {
			next.Next = l1
			l1 = l1.Next
		} else if l1.D <= l2.D {
			next.Next = l1
			l1 = l1.Next
		} else {
			next.Next = l2
			l2 = l2.Next
		}
		next = next.Next
	}

	return root
}

func isAllNullKList(lists []*ListNode) bool {
	for _, l := range lists {
		if l != nil {
			return false
		}
	}

	return true
}

func minIdxKList(lists []*ListNode) int {
	var min *int
	minIdx := -1
	for idx, l := range lists {
		if l != nil && (min == nil || l.D <= *min) {
			min = &l.D
			minIdx = idx
		}
	}

	return minIdx
}

func mergeKLists(lists []*ListNode) *ListNode {
	if isAllNullKList(lists) {
		return nil
	}
	var root *ListNode
	var next *ListNode

	minIdx := minIdxKList(lists)
	root = lists[minIdx]
	next = root
	lists[minIdx] = lists[minIdx].Next
	for {
		if isAllNullKList(lists) {
			break
		}
		minIdx = minIdxKList(lists)
		next.Next = lists[minIdx]
		lists[minIdx] = lists[minIdx].Next

		next = next.Next
	}

	return root
}

func swapPairs(head *ListNode) *ListNode {
	node := head
	var res *ListNode
	var prev *ListNode
	for {
		if node == nil || node.Next == nil {
			if res == nil {
				res = node
			}
			break
		}
		first := node
		second := node.Next
		node = second
		first.Next = second.Next
		node.Next = first
		if res == nil {
			res = node
		}
		if prev != nil {
			prev.Next = node
		}
		prev = first

		node = node.Next.Next
	}

	return res
}

func getAllCombination(arr []string) []string {
	res := []string{strings.Join(arr, "")}
	for i := 0; i < len(arr) - 1; i++ {
		for j := i + 1; j < len(arr); j++ {
			tmp := make([]string, len(arr))
			copy(tmp, arr)
			tmp[i], tmp[j] = tmp[j], tmp[i]
			res = append(res, strings.Join(tmp, ""))
		}
	}

	return res
}

func sliceIntToInt(arr []int) int {
	tmp := make([]int, len(arr))
	copy(tmp, arr)
	res := 0
	m := 1
	reverseArrayInt(tmp)
	for _, a := range tmp {
		res += m * a
		m *= 10
	}

	return res
}

func nextPermutation(nums []int) int {
	all := [][]int{nums}

	for i := 0; i < len(nums) - 1; i++ {
		for j := i + 1; j < len(nums); j++ {
			tmp := make([]int, len(nums))
			copy(tmp, nums)
			tmp[i], tmp[j] = tmp[j], tmp[i]
			all = append(all, tmp)
		}
	}

	allIntMap := make(map[int]struct{}, len(all))
	for _, a := range all {
		allIntMap[sliceIntToInt(a)] = struct{}{}
	}
	allInt := make([]int, 0, len(all))
	for k := range allIntMap {
		allInt = append(allInt, k)
	}

	current := sliceIntToInt(nums)
	res := 0
	sort.Ints(allInt)
	for idx, a := range allInt {
		if a == current {
			if idx < len(allInt) - 1 {
				res = allInt[idx + 1]
			}
			break
		}
	}
	if res == 0 {
		res = allInt[0]
	}

	return res
}

func longestValidParentheses(s string) int {
	arr := strings.Split(s, "")
	maxLen := 0
	for i := 0; i < len(arr) - 1; i++ {
		if maxLen >= len(arr) - i {
			break
		}
		tmp := []string{arr[i]}
		for j := i + 1; j < len(arr); j++ {
			if maxLen >= len(arr) + 1 - j {
				fmt.Println(maxLen, len(arr), j)
				//break
			}
			tmp = append(tmp, arr[j])
			if len(tmp) % 2 == 0 && arr[j] == ")" {
				if len(tmp) > maxLen {
					if correctBrackets(strings.Join(tmp, "")) {
						maxLen = len(tmp)
					}
				}
			}
		}
	}

	return maxLen
}

func longestValidParentheses2(s string) int {
	var stack []rune
	var max int
	var curlen int
	for _, p := range s {
		if p == '(' {
			stack = append(stack, p)
		} else if p == ')' {
			if len(stack) == 0 {
				continue
			}
			tmp := stack[len(stack)-1]
			stack = stack[0:len(stack)-1]
			if tmp == '(' {
				curlen++
				if curlen > max {
					max = curlen
				}
			} else {
				curlen = 0
				stack = []rune{}
			}
		}
	}
	if max != 0 {
		max *= 2
	}
	return max
}

func searchInSortRotated(nums []int, target int) int {
	if len(nums) == 0 {
		return -1
	}
	tmp := make([]int, len(nums))
	copy(tmp, nums)
	idx := 0
	for {
		if len(tmp) == 0 {
			break
		}

		mid := len(tmp) / 2
		if target >= tmp[0] && (target <= tmp[mid-1] ||
			(target > tmp[len(tmp) - 1])) {
			tmp = tmp[:mid]
		} else {
			tmp = tmp[mid:]
			idx += mid
		}
		for _, t := range tmp {
			if t == target {
				return idx
			}
			idx++
		}
	}

	return -1
}

func searchRange(nums []int, target int) []int {
	if len(nums) == 0 {
		return []int{-1, -1}
	}
	tmp := make([]int, len(nums))
	copy(tmp, nums)
	idx := 0
	for {
		if len(tmp) == 0 {
			break
		}

		mid := len(tmp) / 2
		if tmp[mid] == target {
			idx += mid
			fmt.Println("=", mid, idx, tmp)
			from, to := idx, idx
			fromL, toL := mid, mid
			for {
				fromL--
				fmt.Println(fromL, "fromL")
				if fromL < 0 {
					break
				}
				if tmp[fromL] != target {
					break
				}
				fmt.Println("++", from)
				from--
			}
			for {
				toL++
				if toL >= len(tmp) {
					break
				}
				if tmp[toL] != target {
					break
				}
				to++
			}

			return []int{from, to}
		} else {
			if target < tmp[mid] {
				tmp = tmp[:mid]
			} else {
				tmp = tmp[mid+1:]
				idx += mid+1
			}
		}
	}

	return []int{-1, -1}
}

func combinationSum(candidates []int, target int) [][]int {
	tmp := make([]int, len(candidates))
	copy(tmp, candidates)


	for i := 0; i < len(tmp); i++ {
		for j := 0; j < len(tmp); j++ {
			for k := 0; k < len(tmp); k++ {

			}
		}
	}

	return nil
}

func longestValidParentheses3(s string) int {
	stack := []int{-1}
	var result int
	for i := 0; i < len(s); i++ {
		fmt.Println(result, "|", i, len(stack), stack)
		if s[i] == ')' && len(stack) > 1 && s[stack[len(stack)-1]] == '(' {
			stack = stack[:len(stack)-1]
			result = MaxInt(result, i-stack[len(stack)-1])
		} else {
			stack = append(stack, i)
		}
	}
	return result
}

func MaxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func firstMissingPositive(nums []int) int {
	var min, midMin, max int
	m := map[int]bool{}
	for _, n := range nums {
		if n < 0 {
			continue
		}
		m[n] = true
		if n == midMin {
			midMin = 0
		}
		if n < min || min == 0 {
			min = n
		}
		if n > max {
			max = n
		}
		if n - 1 > 0 && (midMin == 0 || n - 1 < midMin) {
			_, ok := m[n-1]
			if !ok {
				midMin = n - 1
			}
		}
	}
	fmt.Println(min)
	//for min != 2 {
	// min--
	//}
	if min > 1 {
		if midMin == 0 || min < midMin {
			midMin = min - 1
		}
	}
	if midMin == 0 && max != 0 {
		midMin = max + 1
	}
	if _, ok := m[1]; !ok {
		midMin = 1
	}

	fmt.Println(min, max, midMin)
	return midMin
}

func nextStep(prev, curr [2]int, matrix [][]int) [2]int {
	n := len(matrix)
	res := [2]int{}

	// Первый шаг
	if prev[0] == curr[0] && prev[1] == curr[1] {
		res[0], res[1] = curr[0], curr[1] + 1
	} else if curr[0] == prev[0] {
		// по горизонтали вдижение

		// слева направо
		if prev[1] < curr[1] {
			nextX := curr[1] + 1
			if nextX == n || matrix[curr[0]][nextX] != 0 {
				res[0], res[1] = curr[0] + 1, curr[1]
			} else {
				res[0], res[1] = curr[0], nextX
			}
		} else {
			// справа на лево
			nextX := curr[1] - 1
			if nextX == -1 || matrix[curr[0]][nextX] != 0 {
				res[0], res[1] = curr[0] - 1, curr[1]
			} else {
				res[0], res[1] = curr[0], nextX
			}
		}
	} else {
		// по вертикали движение

		// сверху вниз
		if prev[0] < curr[0] {
			nextY := curr[0] + 1
			if nextY == n || matrix[nextY][curr[1]] != 0 {
				res[0], res[1] = curr[0], curr[1] - 1
			} else {
				res[0], res[1] = nextY, curr[1]
			}
		} else {
			// снизу вверх
			nextY := curr[0] - 1
			if nextY == -1 || matrix[nextY][curr[1]] != 0 {
				res[0], res[1] = curr[0], curr[1] + 1
			} else {
				res[0], res[1] = nextY, curr[1]
			}
		}
	}
	if res[0] == n || res[1] == n || matrix[res[0]][res[1]] != 0 {
		res[0], res[1] = -1, -1
	}

	return res
}

func generateMatrix(n int) [][]int {
	matrix := make([][]int, 0, n)
	if n == 0 {
		return matrix
	}
	for i := 0; i < n; i++ {
		matrix = append(matrix, make([]int, n))
	}

	prev, curr := [2]int{}, [2]int{}
	val := 1
	matrix[curr[0]][curr[1]] = val
	for {
		next := nextStep(prev, curr, matrix)
		if next[0] == -1 && next[1] == -1 {
			break
		}
		val++
		matrix[next[0]][next[1]] = val
		prev = curr
		curr = next
	}

	return matrix
}


func maxSubArray2(nums []int) (max int) {
	var sum int
	for i := 0; i < len(nums); i++ {
		sum = 0
		for j := i; j < len(nums); j++ {
			sum += nums[j]
			if sum > max {
				max = sum
			}
		}
	}
	return
}

func maxSubArray(nums []int) int {
	tmp := make([]int, 0, len(nums))
	sum := 0
	for _, n := range nums {
		sum += n
		tmp = append(tmp, sum)
	}
	maxIdx, max := 0, 0
	for i := 0; i < len(tmp); i++ {
		if tmp[i] > max {
			max = tmp[i]
			maxIdx = i
		}
	}
	sum, max = 0, 0
	for i := maxIdx; i >= 0; i-- {
		sum += nums[i]
		if sum >= max {
			max = sum
		}
	}

	return max
}

func rotateRight(head *ListNode, k int) *ListNode {
	if head == nil {
		return nil
	}

	result := head

	for i := 0; i < k; i++ {
		node := result
		var last *ListNode
		var prev *ListNode
		for {
			if node.Next != nil {
				prev = node
				node = node.Next
			} else {
				last = node
				break
			}
		}
		last.Next = result
		if prev != nil {
			prev.Next = nil
		} else {
			return head
		}
		result = last
	}

	return result
}

func fullJustify(words []string, maxWidth int) []string {
	result := []string{}
	rows := [][]string{}
	lenRow := 0
	for idx, w := range words {
		if len(rows) == 0 || lenRow + len(w) + 1 < maxWidth {
			if len(rows) != 0 {
				rows = append(rows, []string{" "})
				lenRow++
			}
			rows = append(rows, []string{w})
			lenRow += len(w)
		} else {
			for lenRow != maxWidth {
				for i := 0; i < len(rows); i++ {
					if rows[i][0] == " " {
						rows[i] = append(rows[i], " ")
						lenRow++
						if lenRow == maxWidth {
							break
						}
					}
				}
			}
			row := []string{}
			for _, r := range rows {
				row = append(row, strings.Join(r, ""))
			}
			result = append(result, strings.Join(row, ""))

			if idx == len(words) - 1 {
				result = append(result, w)
			} else {
				rows = [][]string{{w}}
				lenRow = len(w)
			}
		}
	}

	lenLastStr := len(result[len(result) - 1])
	if lenLastStr < maxWidth {
		spaces := make([]string, maxWidth - lenLastStr + 1)
		result[len(result) - 1] += strings.Join(spaces, " ")
		fmt.Println("+++")
	}

	return result
}

func minWindow(s string, t string) string {
	arrStr := strings.Split(s, "")
	arrT := strings.Split(t, "")
	mt := map[string]int{}
	for _, a := range arrT {
		if _, ok := mt[a]; ok {
			mt[a]++
		} else {
			mt[a] = 1
		}
	}
	tmpM := cloneMapInt(mt)

	var minWin, length int
	var minSubs, substr []string
	for j := 0; j < len(arrStr); j++ {
		for i := j; i < len(arrStr); i++ {
			char := arrStr[i]
			if _, ok := tmpM[char]; !ok && countRestCharInMap(tmpM) == len(t) {
				continue
			}
			length++
			substr = append(substr, char)
			if v, ok := mt[char]; ok {
				if v > 1 {
					tmpM[char]--
				} else {
					delete(tmpM, char)
				}
			}
			if len(tmpM) == 0 {
				if minWin == 0 || length < minWin {
					minWin = length
					minSubs = substr
					length = 0
				}
				if len(arrStr) - i < minWin {
					break
				}
				tmpM = cloneMapInt(mt)
				substr = []string{}
			}
		}
	}

	return strings.Join(minSubs, "")
}

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func sliceIntToBT(arr []*int) *TreeNode {
	if len(arr) == 0 || arr[0] == nil {
		return nil
	}
	head := &TreeNode{
		Val: *arr[0],
	}
	//prev := head
	prevs := []*TreeNode{head}
	for i := 1; i < len(arr); i += 2 {
		//l := len(prevs)
		prev := prevs[0]
		fmt.Println(prev.Val, i, "++++")
		if v := arr[i]; v != nil {
			left := &TreeNode{
				Val: *v,
			}
			prev.Left = left
			prevs = append(prevs, left)
		}
		if v := arr[i+1]; v != nil {
			right := &TreeNode{
				Val: *v,
			}
			prev.Right = right
			prevs = append(prevs, right)
		}
		for _, p := range prevs {
			fmt.Println(p.Val, "v")
		}
		fmt.Println("====")
		prevs = prevs[1:]
	}

	return head
}

func createNodeBTWithChild(arr []*int, idx *int) *TreeNode {
	if idx == nil || *idx < 0 || *idx >= len(arr) || arr[*idx] == nil {
		return nil
	}
	node := &TreeNode {
		Val: *arr[*idx],
	}

	*idx++
	if *idx >= len(arr) {
		return node
	}
	left := &TreeNode{}
	if v := arr[*idx]; v != nil {
		left.Val = *v
		node.Left = left
	}

	*idx++
	if *idx >= len(arr) {
		return node
	}
	right := &TreeNode{}
	if v := arr[*idx]; v != nil {
		right.Val = *v
		node.Right = right
	}

	if node.Left != nil {
		*idx++
		left.Left = createNodeBTWithChild(arr, idx)
		*idx++
		left.Right = createNodeBTWithChild(arr, idx)
	}

	if node.Right != nil {
		*idx++
		right.Left = createNodeBTWithChild(arr, idx)
		*idx++
		right.Right = createNodeBTWithChild(arr, idx)
	}

	return node
}

func isValidBST(root *TreeNode) bool {
	if root == nil {
		return false
	}
	left, right := []int{root.Val}, []int{root.Val}

	collectLeftOrRight(root.Left, &left)
	for idx, l := range left {
		if idx > 0 && l > left[idx-1] {
			return false
		}
	}

	collectLeftOrRight(root.Right, &right)
	for idx, r := range right {
		if idx > 0 && r < left[idx-1] {
			return false
		}
	}

	return true
}

func getSortedSliceByBST(root *TreeNode, arr *[]int) {
	if root == nil {
		return
	}
	if root.Left == nil && root.Right == nil {
		*arr = append(*arr, root.Val)
		return
	}
	if root.Left != nil {
		collectLeftOrRight(root.Left, arr)
	}

	*arr = append(*arr, root.Val)

	if root.Right != nil {
		collectLeftOrRight(root.Right, arr)
	}
}

func collectLeftOrRight(root *TreeNode, arr *[]int) {
	if root == nil {
		return
	}
	if root.Left == nil && root.Right == nil {
		*arr = append(*arr, root.Val)
		return
	}
	if root.Left != nil {
		collectLeftOrRight(root.Left, arr)
	}

	if root.Right != nil {
		collectLeftOrRight(root.Right, arr)
	}

	*arr = append(*arr, root.Val)
}

func traverBT(root *TreeNode) {
	if root == nil {
		return
	}
	if root.Left != nil || root.Right != nil {
		fmt.Println("root", root.Val)
	}

	if root.Left != nil {
		fmt.Println("left", root.Left.Val)
		traverBT(root.Left)
	}
	if root.Right != nil {
		fmt.Println("right", root.Right.Val)
		traverBT(root.Right)
	}
}

func countRestCharInMap(mt map[string]int) int {
	i := 0
	for _, v := range mt {
		i += v
	}

	return i
}

func cloneMapInt(from map[string]int) map[string]int {
	to := make(map[string]int, len(from))
	for k, v := range from {
		to[k] = v
	}

	return to
}

func i(n int) *int {
	return &n
}

func app(arr *[]int) {
	*arr = append(*arr, 1)
}

type sliceToBST struct {
	idx int
	val int
	isNil bool
	left []int
	right []int
}

func sortedArrayToBST(nums []int) *TreeNode {
	result := make([]*int, 0, int(float64(len(nums)) * 1.5))
	mid := len(nums) / 2
	intermediateResult := make([]sliceToBST, 0, len(nums) / 2)

	intermediateResult = append(intermediateResult, sliceToBST{
		idx: mid,
		val: nums[mid],
		left: nums[:mid],
		right: nums[mid+1:],
	})

	for {
		l := len(intermediateResult)
		if l == 0 {
			break
		}
		for i := 0; i < l; i++ {
			res := intermediateResult[i]
			if res.isNil {
				result = append(result, nil)
			} else {
				val := res.val
				result = append(result, &val)
			}
			if len(res.left) > 0 {
				mid := len(res.left) / 2
				intermediateResult = append(intermediateResult, sliceToBST{
					idx: mid,
					val: res.left[mid],
					left: res.left[:mid],
					right: res.left[mid+1:],
				})
			}
			if len(res.right) > 0 {
				mid := len(res.right) / 2
				intermediateResult = append(intermediateResult, sliceToBST{
					idx: mid,
					val: res.right[mid],
					left: res.right[:mid],
					right: res.right[mid+1:],
				})
			} else if len(res.left) > 0 {
				intermediateResult = append(intermediateResult, sliceToBST{
					isNil: true,
				})
			}
		}
		intermediateResult = intermediateResult[l:]
	}

	for _, r := range result {
		if r == nil {
			fmt.Println("nil")
		} else {
			fmt.Println(*r)
		}
	}

	return sliceIntToBT(result)
}

/*
func sortedArrayToSlice(nums []int, result []int, idx *int) int {
	if len(nums) == 0 {
		return -1
	} else if len(nums) == 1 {
		result = append(result, nums[0])
		*idx++
		return -1
	}
	/*
	else if len(nums) == 2 {
		fmt.Println("=====")
		*result = append(*result, nums[0], nums[1])
		fmt.Println("=====", result)
		return
	}
	*/
/*
	mid := len(nums) / 2
	//if len(nums) % 2 == 0 {
	//	mid--
	//}
	fmt.Println("---", nums[mid], mid)
	result = append(result, nums[mid])
	//fmt.Println(result)
	left := nums[:mid]
	sortedArrayToSlice(left, result, )
	right := nums[mid+1:]
	//sortedArrayToSlice(right, result)

	return mid
}
*/

func appS(arr []int) {
	arr = append(arr, 1)
}

func comparePalindromePart(left, right []string) bool {
	for i := 0; i < len(left); i++ {
		if left[len(left) - 1 - i] != right[i] {
			return false
		}
	}

	return true
}

func shortestPalindrome(s string) string {
	arr := strings.Split(s, "")
	mid := len(arr) / 2
	left := arr[:mid]
	right := arr[mid:]
	midString := ""
	for i := mid - 1; i >= 0; i-- {
		if comparePalindromePart(left, right) {
			break
		}
		if len(right) > len(left) {
			if comparePalindromePart(left, right[1:]) {
				midString = right[0]
				right = right[1:]
				break
			}
		}
		left = left[:len(left)-1]
		right = append([]string{arr[i]}, right...)
	}

	if len(left) == 0 {
		midString = right[0]
		right = right[1:]
	}

	for i := len(left); i < len(right); i++ {
		left = append([]string{right[i]}, left...)
	}

	res := make([]string, len(left) + len(right) + 1)
	res = append(res, left...)
	res = append(res, midString)
	res = append(res, right...)

	return strings.Join(res, "")
}

func main() {
	fmt.Println(shortestPalindrome("aacecaaa"))
	fmt.Println(shortestPalindrome("abcd"))
	/*
	fmt.Println(len("Привет!"), "Привет!")
	arr := []int{1,3,5,7,8,9}
	for _, a := range arr {
		fmt.Println(a)
		//if idx ==
		arr = append(arr, 0)
	}

	fmt.Println(arr)

	tree := sortedArrayToBST([]int{-10,-3,0,5,9})
	traverBT(tree)

	//tree2 := sortedArrayToBST([]int{-50,-40,30,20,-10,-3,0,5,9,10,20,30,40,50})
	//(tree2)

	//sliceIntToBT([]int{5,1,4,nil,nil,3,6})
	//for _, a := range []*int{i(5),i(1),i(4),nil,nil,i(3),i(6)} {
	//	fmt.Println(a)
	//}
	//idx := 0
	//tree := createNodeBTWithChild([]*int{i(2),i(1),i(3)}, &idx)
	//idx2 := 0
	//tree2 := createNodeBTWithChild([]*int{i(5),i(1),i(4),nil,nil,i(3),i(6)}, &idx2)
	/*
	idx3 := 0
	tree3 := createNodeBTWithChild([]*int{i(3),i(1),i(4),nil,nil,i(2)}, &idx3)
	idx4 := 0
	tree4 := createNodeBTWithChild([]*int{i(2),i(1),i(4),nil,nil,i(3)}, &idx4)

	//fmt.Println(isValidBST(tree))
	//fmt.Println(isValidBST(tree2))
	fmt.Println(isValidBST(tree3), "valid")
	fmt.Println(isValidBST(tree4), "valid")
	//res := createNodeBTWithChild([]*int{i(5),i(1),i(4),nil,nil,i(3),i(6)}, &idx)
	traverBT(tree3)
	traverBT(tree4)

	var sorted []int
	getSortedSliceByBST(tree4, &sorted)
	fmt.Println(sorted)

	//fmt.Println(minWindow("ADOBECODEBANC", "ABC"))
	/*
	res := fullJustify([]string{"This", "is", "an", "example", "of", "text", "justification."}, 16)
	for _, r := range res {
		fmt.Println(r, len(r))
	}

	 */

	//fmt.Println(maxSubArray([]int{-2,1,5,7, -100, 200}))
	//fmt.Println(maxSubArray2([]int{-2,1,-3,4,-1,2,1,-5,4}))

	//fmt.Println(searchInSortRotated([]int{6,7,0,1,2,4,5}, 4))

	//fmt.Println(longestValidParentheses3("(()(()(())()"))
	//fmt.Println(longestValidParentheses( "(()(()(())()"))

	//fmt.Println(longestValidParentheses2("(())(()"))
	//fmt.Println(longestValidParentheses("(())(()"))
	//fmt.Println(nextPermutation([]int{1,1,5}))

	//fmt.Println(6 << 4)
	/*
	   lists := []*ListNode{
	      arr2ListNode([]int{1,2,4, 9}),
	      arr2ListNode([]int{1,3,4}),
	      arr2ListNode([]int{2,6}),
	   }

	   res := swapPairs(arr2ListNode([]int{1,2,3,4,5}))

	   for {
	      if res == nil {
	         break
	      }
	      fmt.Println(res.D, "=")
	      res = res.Next
	   }


	   /*
	   str := "fjdfkwdwediezxgdsfnwnelcnelcnfewekfnerjkfu"
	   arr := strings.Split(str, "")
	   l := len(arr)
	   fmt.Println(arr)

	   maxStr := []string{}
	   tmpMax := []string{}
	   to := 0
	   needAppend := true
	   for to < l {
	      if needAppend {
	         //fmt.Println(to)
	         tmpMax = append(tmpMax, arr[to])
	         to = to + 1
	         if isUnique(tmpMax) {
	            if len(tmpMax) > len(maxStr) {
	               maxStr = tmpMax
	            }
	         } else {
	            needAppend = false
	         }
	      } else {
	         tmpMax = tmpMax[1:]
	         //fmt.Println(tmpMax)
	         if isUnique(tmpMax) {
	            needAppend = true
	         }
	      }

	   }

	   fmt.Println(maxStr)
	*/
}