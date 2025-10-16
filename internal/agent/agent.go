package agent

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"

	"agent_challenge/internal/openrouter"
)

// GetToolDefinitions returns OpenAI-compatible tool definitions
func GetToolDefinitions() []openrouter.Tool {
	return []openrouter.Tool{
		{
			Type: "function",
			Function: openrouter.ToolFunction{
				Name:        "get_time",
				Description: "Возвращает текущее время в формате RFC3339 (UTC)",
				Parameters: map[string]any{
					"type":       "object",
					"properties": map[string]any{},
				},
			},
		},
		{
			Type: "function",
			Function: openrouter.ToolFunction{
				Name:        "calc",
				Description: "Вычисляет арифметическое выражение (+,-,*,/, скобки)",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"expression": map[string]any{
							"type":        "string",
							"description": "Арифметическое выражение",
						},
					},
					"required": []string{"expression"},
				},
			},
		},
	}
}

func ExecuteTool(tc openrouter.ToolCall) string {
	switch tc.Function.Name {
	case "get_time":
		return time.Now().UTC().Format(time.RFC3339)
	case "calc":
		var args struct {
			Expression string `json:"expression"`
		}
		_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
		res, err := evalExpression(args.Expression)
		if err != nil {
			return "error: " + err.Error()
		}
		return res
	default:
		return "error: unknown tool"
	}
}

// evalExpression evaluates a simple arithmetic expression with + - * / and parentheses.
// Returns string result (trimmed trailing zeros) or error.
func evalExpression(expr string) (string, error) {
	// Tokenize
	tokens, err := tokenize(expr)
	if err != nil {
		return "", err
	}
	// Convert to RPN via shunting-yard
	rpn, err := toRPN(tokens)
	if err != nil {
		return "", err
	}
	// Evaluate RPN
	val, err := evalRPN(rpn)
	if err != nil {
		return "", err
	}
	// Format number removing trailing zeros
	res := strconv.FormatFloat(val, 'f', -1, 64)
	if strings.Contains(res, "e+") || strings.Contains(res, "e-") {
		res = strconv.FormatFloat(val, 'f', 10, 64)
		res = strings.TrimRight(strings.TrimRight(res, "0"), ".")
	}
	return res, nil
}

func tokenize(s string) ([]string, error) {
	s = strings.TrimSpace(s)
	var tokens []string
	n := len(s)
	for i := 0; i < n; {
		ch := s[i]
		switch {
		case ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r':
			i++
		case ch == '+' || ch == '-' || ch == '*' || ch == '/' || ch == '(' || ch == ')':
			// handle unary minus: if at start or after '(' or operator, treat as part of number
			if ch == '-' {
				if i == 0 || (len(tokens) > 0 && (tokens[len(tokens)-1] == "(" || isOperator(tokens[len(tokens)-1]))) {
					// negative number
					j := i + 1
					seenDot := false
					for j < n {
						c := s[j]
						if c >= '0' && c <= '9' {
							j++
							continue
						}
						if c == '.' {
							if seenDot {
								break
							}
							seenDot = true
							j++
							continue
						}
						break
					}
					if j > i+1 {
						tokens = append(tokens, s[i:j])
						i = j
						continue
					}
				}
			}
			tokens = append(tokens, string(ch))
			i++
		case (ch >= '0' && ch <= '9') || ch == '.':
			j := i
			seenDot := false
			for j < n {
				c := s[j]
				if c >= '0' && c <= '9' {
					j++
					continue
				}
				if c == '.' {
					if seenDot {
						break
					}
					seenDot = true
					j++
					continue
				}
				break
			}
			if j == i || (j == i+1 && s[i] == '.') {
				return nil, fmt.Errorf("invalid number at %d", i)
			}
			tokens = append(tokens, s[i:j])
			i = j
		default:
			return nil, fmt.Errorf("invalid character: %q", ch)
		}
	}
	return tokens, nil
}

func precedence(op string) int {
	switch op {
	case "+", "-":
		return 1
	case "*", "/":
		return 2
	default:
		return 0
	}
}

func isOperator(tok string) bool {
	return tok == "+" || tok == "-" || tok == "*" || tok == "/"
}

func toRPN(tokens []string) ([]string, error) {
	var output []string
	var stack []string
	for _, tok := range tokens {
		switch {
		case isOperator(tok):
			for len(stack) > 0 && isOperator(stack[len(stack)-1]) && precedence(stack[len(stack)-1]) >= precedence(tok) {
				output = append(output, stack[len(stack)-1])
				stack = stack[:len(stack)-1]
			}
			stack = append(stack, tok)
		case tok == "(":
			stack = append(stack, tok)
		case tok == ")":
			found := false
			for len(stack) > 0 {
				t := stack[len(stack)-1]
				stack = stack[:len(stack)-1]
				if t == "(" {
					found = true
					break
				}
				output = append(output, t)
			}
			if !found {
				return nil, fmt.Errorf("mismatched parentheses")
			}
		default:
			// number
			output = append(output, tok)
		}
	}
	for i := len(stack) - 1; i >= 0; i-- {
		if stack[i] == "(" || stack[i] == ")" {
			return nil, fmt.Errorf("mismatched parentheses")
		}
		output = append(output, stack[i])
	}
	return output, nil
}

func evalRPN(rpn []string) (float64, error) {
	var st []float64
	for _, tok := range rpn {
		if isOperator(tok) {
			if len(st) < 2 {
				return 0, fmt.Errorf("invalid expression")
			}
			b := st[len(st)-1]
			st = st[:len(st)-1]
			a := st[len(st)-1]
			st = st[:len(st)-1]
			switch tok {
			case "+":
				st = append(st, a+b)
			case "-":
				st = append(st, a-b)
			case "*":
				st = append(st, a*b)
			case "/":
				if b == 0 {
					return 0, fmt.Errorf("division by zero")
				}
				st = append(st, a/b)
			}
			continue
		}
		// number
		v, err := strconv.ParseFloat(tok, 64)
		if err != nil {
			return 0, err
		}
		if math.IsInf(v, 0) || math.IsNaN(v) {
			return 0, fmt.Errorf("invalid number")
		}
		st = append(st, v)
	}
	if len(st) != 1 {
		return 0, fmt.Errorf("invalid expression")
	}
	return st[0], nil
}
