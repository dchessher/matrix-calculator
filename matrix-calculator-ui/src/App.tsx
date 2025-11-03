import { useEffect, useMemo, useState } from 'react'
import './App.css'

type MatrixInput = string[][]

type NumericMatrix = number[][]

type Operation =
  | 'add'
  | 'subtract'
  | 'multiply'
  | 'hadamard'
  | 'detA'
  | 'detB'
  | 'invA'
  | 'invB'
  | 'transA'
  | 'transB'

type ResultState =
  | { type: 'matrix'; label: string; data: NumericMatrix }
  | { type: 'scalar'; label: string; value: number }
  | { type: 'message'; label: string; message: string }

type DimensionSet = {
  rowsA: number
  colsA: number
  rowsB: number
  colsB: number
}

type OperationConfig = {
  value: Operation
  label: string
  description: string
  guard: (dims: DimensionSet) => boolean
}

const DIMENSION_OPTIONS = Array.from({ length: 6 }, (_, index) => index + 1)

const OPERATION_CONFIG: OperationConfig[] = [
  {
    value: 'add',
    label: 'A + B',
    description: 'Element-wise addition of the two matrices.',
    guard: ({ rowsA, colsA, rowsB, colsB }) => rowsA === rowsB && colsA === colsB,
  },
  {
    value: 'subtract',
    label: 'A - B',
    description: 'Element-wise subtraction of matrix B from matrix A.',
    guard: ({ rowsA, colsA, rowsB, colsB }) => rowsA === rowsB && colsA === colsB,
  },
  {
    value: 'hadamard',
    label: 'A ⊙ B (Hadamard)',
    description: 'Element-wise multiplication, useful for feature scaling and gating.',
    guard: ({ rowsA, colsA, rowsB, colsB }) => rowsA === rowsB && colsA === colsB,
  },
  {
    value: 'multiply',
    label: 'A × B',
    description: 'Matrix multiplication, core to linear transformations.',
    guard: ({ colsA, rowsB }) => colsA === rowsB,
  },
  {
    value: 'detA',
    label: 'det(A)',
    description: 'Determinant of matrix A — only defined for square matrices.',
    guard: ({ rowsA, colsA }) => rowsA === colsA,
  },
  {
    value: 'detB',
    label: 'det(B)',
    description: 'Determinant of matrix B — only defined for square matrices.',
    guard: ({ rowsB, colsB }) => rowsB === colsB,
  },
  {
    value: 'invA',
    label: 'A⁻¹',
    description: 'Inverse of matrix A when it is non-singular and square.',
    guard: ({ rowsA, colsA }) => rowsA === colsA,
  },
  {
    value: 'invB',
    label: 'B⁻¹',
    description: 'Inverse of matrix B when it is non-singular and square.',
    guard: ({ rowsB, colsB }) => rowsB === colsB,
  },
  {
    value: 'transA',
    label: 'Aᵀ',
    description: 'Transpose of matrix A — swap rows with columns.',
    guard: () => true,
  },
  {
    value: 'transB',
    label: 'Bᵀ',
    description: 'Transpose of matrix B — swap rows with columns.',
    guard: () => true,
  },
]

function createMatrix(rows: number, cols: number, previous?: MatrixInput): MatrixInput {
  return Array.from({ length: rows }, (_, rowIndex) =>
    Array.from({ length: cols }, (_, colIndex) => previous?.[rowIndex]?.[colIndex] ?? '0'),
  )
}

function parseMatrix(matrix: MatrixInput): NumericMatrix {
  return matrix.map((row) =>
    row.map((value) => {
      if (value.trim() === '') return 0
      const parsed = Number(value)
      return Number.isFinite(parsed) ? parsed : 0
    }),
  )
}

function addMatrices(a: NumericMatrix, b: NumericMatrix): NumericMatrix {
  return a.map((row, i) => row.map((value, j) => value + b[i][j]))
}

function subtractMatrices(a: NumericMatrix, b: NumericMatrix): NumericMatrix {
  return a.map((row, i) => row.map((value, j) => value - b[i][j]))
}

function hadamardProduct(a: NumericMatrix, b: NumericMatrix): NumericMatrix {
  return a.map((row, i) => row.map((value, j) => value * b[i][j]))
}

function multiplyMatrices(a: NumericMatrix, b: NumericMatrix): NumericMatrix {
  const rows = a.length
  const cols = b[0]?.length ?? 0
  const shared = a[0]?.length ?? 0
  const result: NumericMatrix = Array.from({ length: rows }, () => Array(cols).fill(0))

  for (let i = 0; i < rows; i += 1) {
    for (let j = 0; j < cols; j += 1) {
      let sum = 0
      for (let k = 0; k < shared; k += 1) {
        sum += a[i][k] * b[k][j]
      }
      result[i][j] = sum
    }
  }

  return result
}

function transpose(matrix: NumericMatrix): NumericMatrix {
  const rows = matrix.length
  const cols = matrix[0]?.length ?? 0
  return Array.from({ length: cols }, (_, col) =>
    Array.from({ length: rows }, (_, row) => matrix[row][col]),
  )
}

function determinant(matrix: NumericMatrix): number {
  const n = matrix.length
  if (n === 0 || matrix.some((row) => row.length !== n)) {
    throw new Error('Determinant is only defined for non-empty square matrices.')
  }

  const temp = matrix.map((row) => [...row])
  let det = 1
  let sign = 1

  for (let col = 0; col < n; col += 1) {
    let pivot = col
    for (let row = col; row < n; row += 1) {
      if (Math.abs(temp[row][col]) > Math.abs(temp[pivot][col])) {
        pivot = row
      }
    }

    const pivotValue = temp[pivot][col]
    if (Math.abs(pivotValue) < 1e-10) {
      return 0
    }

    if (pivot !== col) {
      ;[temp[pivot], temp[col]] = [temp[col], temp[pivot]]
      sign *= -1
    }

    det *= temp[col][col]

    for (let row = col + 1; row < n; row += 1) {
      const factor = temp[row][col] / temp[col][col]
      for (let k = col; k < n; k += 1) {
        temp[row][k] -= factor * temp[col][k]
      }
    }
  }

  return det * sign
}

function inverse(matrix: NumericMatrix): NumericMatrix {
  const n = matrix.length
  if (n === 0 || matrix.some((row) => row.length !== n)) {
    throw new Error('Inverse is only defined for non-empty square matrices.')
  }

  const augmented = matrix.map((row, i) => [
    ...row,
    ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  ])

  for (let col = 0; col < n; col += 1) {
    let pivot = col
    for (let row = col; row < n; row += 1) {
      if (Math.abs(augmented[row][col]) > Math.abs(augmented[pivot][col])) {
        pivot = row
      }
    }

    const pivotValue = augmented[pivot][col]
    if (Math.abs(pivotValue) < 1e-10) {
      throw new Error('Matrix is singular and cannot be inverted.')
    }

    if (pivot !== col) {
      ;[augmented[pivot], augmented[col]] = [augmented[col], augmented[pivot]]
    }

    for (let j = 0; j < 2 * n; j += 1) {
      augmented[col][j] /= pivotValue
    }

    for (let row = 0; row < n; row += 1) {
      if (row === col) continue
      const factor = augmented[row][col]
      for (let j = 0; j < 2 * n; j += 1) {
        augmented[row][j] -= factor * augmented[col][j]
      }
    }
  }

  return augmented.map((row) => row.slice(n))
}

function formatNumber(value: number): string {
  if (Math.abs(value) < 1e-10) return '0'
  const rounded = Number(value.toFixed(6))
  return Math.abs(rounded) < 1e-10 ? '0' : rounded.toString()
}

type MatrixEditorProps = {
  name: 'A' | 'B'
  rows: number
  cols: number
  data: MatrixInput
  onRowsChange: (rows: number) => void
  onColsChange: (cols: number) => void
  onCellChange: (row: number, col: number, value: string) => void
}

function MatrixEditor({ name, rows, cols, data, onRowsChange, onColsChange, onCellChange }: MatrixEditorProps) {
  return (
    <div className="matrix-card">
      <div className="matrix-header">
        <div>
          <h2>Matrix {name}</h2>
          <p className="matrix-subtitle">Populate values or paste from your workflow.</p>
        </div>
        <div className="dimension-controls">
          <label className="dimension-control">
            <span>Rows</span>
            <select value={rows} onChange={(event) => onRowsChange(Number(event.target.value))}>
              {DIMENSION_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <label className="dimension-control">
            <span>Cols</span>
            <select value={cols} onChange={(event) => onColsChange(Number(event.target.value))}>
              {DIMENSION_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>
      <div
        className="matrix-grid"
        style={{ gridTemplateColumns: `repeat(${cols}, minmax(3.5rem, 1fr))` }}
        aria-label={`Matrix ${name} grid`}
      >
        {data.map((row, rowIndex) =>
          row.map((value, colIndex) => (
            <input
              key={`${rowIndex}-${colIndex}`}
              type="number"
              inputMode="decimal"
              step="any"
              value={value}
              onChange={(event) => onCellChange(rowIndex, colIndex, event.target.value)}
              aria-label={`Matrix ${name} cell ${rowIndex + 1}, ${colIndex + 1}`}
            />
          )),
        )}
      </div>
      <div className="matrix-footnote">Currently {rows} × {cols}</div>
    </div>
  )
}

function App() {
  const [rowsA, setRowsA] = useState(2)
  const [colsA, setColsA] = useState(2)
  const [rowsB, setRowsB] = useState(2)
  const [colsB, setColsB] = useState(2)

  const [matrixA, setMatrixA] = useState<MatrixInput>(createMatrix(2, 2))
  const [matrixB, setMatrixB] = useState<MatrixInput>(createMatrix(2, 2))

  const [operation, setOperation] = useState<Operation>('add')
  const [result, setResult] = useState<ResultState | null>(null)

  const dimensions = useMemo<DimensionSet>(
    () => ({ rowsA, colsA, rowsB, colsB }),
    [rowsA, colsA, rowsB, colsB],
  )

  const availableOperations = useMemo(
    () => OPERATION_CONFIG.filter((config) => config.guard(dimensions)),
    [dimensions],
  )

  useEffect(() => {
    if (!availableOperations.some((config) => config.value === operation)) {
      setOperation(availableOperations[0]?.value ?? 'transA')
    }
  }, [availableOperations, operation])

  const selectedOperation = availableOperations.find((config) => config.value === operation)

  const handleRowsChange = (matrix: 'A' | 'B') => (value: number) => {
    if (matrix === 'A') {
      setRowsA(value)
      setMatrixA((prev) => createMatrix(value, colsA, prev))
    } else {
      setRowsB(value)
      setMatrixB((prev) => createMatrix(value, colsB, prev))
    }
  }

  const handleColsChange = (matrix: 'A' | 'B') => (value: number) => {
    if (matrix === 'A') {
      setColsA(value)
      setMatrixA((prev) => createMatrix(rowsA, value, prev))
    } else {
      setColsB(value)
      setMatrixB((prev) => createMatrix(rowsB, value, prev))
    }
  }

  const handleCellChange = (matrix: 'A' | 'B') => (row: number, col: number, value: string) => {
    if (matrix === 'A') {
      setMatrixA((prev) => {
        const next = prev.map((line) => [...line])
        next[row][col] = value
        return next
      })
    } else {
      setMatrixB((prev) => {
        const next = prev.map((line) => [...line])
        next[row][col] = value
        return next
      })
    }
  }

  const handleCalculate = () => {
    const numericA = parseMatrix(matrixA)
    const numericB = parseMatrix(matrixB)

    try {
      switch (operation) {
        case 'add':
          setResult({ type: 'matrix', label: 'A + B', data: addMatrices(numericA, numericB) })
          break
        case 'subtract':
          setResult({ type: 'matrix', label: 'A - B', data: subtractMatrices(numericA, numericB) })
          break
        case 'hadamard':
          setResult({ type: 'matrix', label: 'A ⊙ B', data: hadamardProduct(numericA, numericB) })
          break
        case 'multiply':
          setResult({ type: 'matrix', label: 'A × B', data: multiplyMatrices(numericA, numericB) })
          break
        case 'detA':
          setResult({ type: 'scalar', label: 'det(A)', value: determinant(numericA) })
          break
        case 'detB':
          setResult({ type: 'scalar', label: 'det(B)', value: determinant(numericB) })
          break
        case 'invA':
          setResult({ type: 'matrix', label: 'A⁻¹', data: inverse(numericA) })
          break
        case 'invB':
          setResult({ type: 'matrix', label: 'B⁻¹', data: inverse(numericB) })
          break
        case 'transA':
          setResult({ type: 'matrix', label: 'Aᵀ', data: transpose(numericA) })
          break
        case 'transB':
          setResult({ type: 'matrix', label: 'Bᵀ', data: transpose(numericB) })
          break
        default:
          setResult({ type: 'message', label: 'Notice', message: 'Select a valid operation.' })
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'An unexpected error occurred.'
      setResult({ type: 'message', label: 'Calculation error', message })
    }
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>Matrix Lab for Machine Learning</h1>
          <p>
            Tune your matrices, explore transformations, and see compatible operations update in
            real-time.
          </p>
        </div>
        <div className="operation-chip">Dark mode · Optimized for clarity</div>
      </header>

      <section className="matrix-section">
        <MatrixEditor
          name="A"
          rows={rowsA}
          cols={colsA}
          data={matrixA}
          onRowsChange={handleRowsChange('A')}
          onColsChange={handleColsChange('A')}
          onCellChange={handleCellChange('A')}
        />
        <MatrixEditor
          name="B"
          rows={rowsB}
          cols={colsB}
          data={matrixB}
          onRowsChange={handleRowsChange('B')}
          onColsChange={handleColsChange('B')}
          onCellChange={handleCellChange('B')}
        />
      </section>

      <section className="operation-section">
        <div className="operation-panel">
          <div className="operation-heading">
            <h2>Choose a computation</h2>
            <p>Select an operation available for the current matrix dimensions.</p>
          </div>
          <div className="operation-controls">
            <label className="operation-select">
              <span>Operation</span>
              <select
                value={operation}
                onChange={(event) => setOperation(event.target.value as Operation)}
              >
                {availableOperations.map((config) => (
                  <option key={config.value} value={config.value}>
                    {config.label}
                  </option>
                ))}
              </select>
            </label>
            <button type="button" onClick={handleCalculate} className="calculate-button">
              Calculate
            </button>
          </div>
          {selectedOperation && (
            <p className="operation-description">{selectedOperation.description}</p>
          )}
        </div>
      </section>

      {result && (
        <section className="result-section">
          <div className="result-card">
            <div className="result-header">
              <h3>{result.label}</h3>
              <span className="result-chip">Output</span>
            </div>
            {result.type === 'matrix' && result.data.length > 0 ? (
              <div
                className="matrix-grid result-grid"
                style={{
                  gridTemplateColumns: `repeat(${result.data[0]?.length ?? 0}, minmax(3.5rem, 1fr))`,
                }}
                aria-live="polite"
              >
                {result.data.map((row, rowIndex) =>
                  row.map((value, colIndex) => (
                    <div key={`${rowIndex}-${colIndex}`} className="result-cell">
                      {formatNumber(value)}
                    </div>
                  )),
                )}
              </div>
            ) : null}
            {result?.type === 'matrix' && result.data.length === 0 ? (
              <p className="result-message">Empty matrix</p>
            ) : null}
            {result?.type === 'scalar' ? (
              <div className="scalar-result">{formatNumber(result.value)}</div>
            ) : null}
            {result?.type === 'message' ? (
              <p className="result-message">{result.message}</p>
            ) : null}
          </div>
        </section>
      )}
    </div>
  )
}

export default App
