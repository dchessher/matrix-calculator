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

type VisualizationMatrix = {
  label: string
  data: NumericMatrix
}

type VisualizationScalar = {
  label: string
  value: number
}

type VisualizationStep = {
  title: string
  description?: string
  matrices?: VisualizationMatrix[]
  scalar?: VisualizationScalar
}

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

function cloneMatrix(matrix: NumericMatrix): NumericMatrix {
  return matrix.map((row) => [...row])
}

function calculateDeterminantWithSteps(matrix: NumericMatrix): {
  value: number
  steps: VisualizationStep[]
} {
  const n = matrix.length
  if (n === 0 || matrix.some((row) => row.length !== n)) {
    throw new Error('Determinant is only defined for non-empty square matrices.')
  }

  const working = cloneMatrix(matrix)
  let det = 1
  let sign = 1
  const steps: VisualizationStep[] = [
    {
      title: 'Start with matrix',
      description: 'Use Gaussian elimination to transform the matrix to upper-triangular form.',
      matrices: [{ label: 'Initial matrix', data: cloneMatrix(working) }],
    },
  ]

  for (let col = 0; col < n; col += 1) {
    let pivot = col
    for (let row = col; row < n; row += 1) {
      if (Math.abs(working[row][col]) > Math.abs(working[pivot][col])) {
        pivot = row
      }
    }

    const pivotValue = working[pivot][col]
    if (Math.abs(pivotValue) < 1e-10) {
      steps.push({
        title: `Column ${col + 1}`,
        description:
          'No non-zero pivot was found in this column, so the determinant collapses to 0.',
      })
      return { value: 0, steps }
    }

    if (pivot !== col) {
      ;[working[pivot], working[col]] = [working[col], working[pivot]]
      sign *= -1
      steps.push({
        title: `Swap rows ${col + 1} and ${pivot + 1}`,
        description: 'Swapping rows flips the determinant sign.',
        matrices: [{ label: 'After row swap', data: cloneMatrix(working) }],
      })
    }

    det *= working[col][col]

    for (let row = col + 1; row < n; row += 1) {
      const factor = working[row][col] / working[col][col]
      if (Math.abs(factor) < 1e-10) continue
      for (let k = col; k < n; k += 1) {
        working[row][k] -= factor * working[col][k]
      }
    }

    steps.push({
      title: `Eliminate column ${col + 1}`,
      description: `Zero out entries below the pivot in column ${col + 1}.`,
      matrices: [{ label: 'Upper-triangular form', data: cloneMatrix(working) }],
    })
  }

  const value = det * sign
  steps.push({
    title: 'Multiply diagonal entries',
    description: 'Multiply the diagonal entries and apply the accumulated sign to get det(A).',
    scalar: { label: 'determinant', value },
  })

  return { value, steps }
}

function calculateInverseWithSteps(matrix: NumericMatrix): {
  value: NumericMatrix
  steps: VisualizationStep[]
} {
  const n = matrix.length
  if (n === 0 || matrix.some((row) => row.length !== n)) {
    throw new Error('Inverse is only defined for non-empty square matrices.')
  }

  const augmented = matrix.map((row, i) => [
    ...row,
    ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  ])

  const steps: VisualizationStep[] = [
    {
      title: 'Augment with identity',
      description: 'Start Gauss-Jordan elimination on the augmented matrix [A | I].',
      matrices: [{ label: '[A | I]', data: cloneMatrix(augmented) }],
    },
  ]

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
      steps.push({
        title: `Swap rows ${col + 1} and ${pivot + 1}`,
        description: 'Bring a strong pivot into position to maintain numerical stability.',
        matrices: [{ label: 'After row swap', data: cloneMatrix(augmented) }],
      })
    }

    const currentPivot = augmented[col][col]
    if (Math.abs(currentPivot - 1) > 1e-10) {
      for (let j = 0; j < 2 * n; j += 1) {
        augmented[col][j] /= currentPivot
      }
      steps.push({
        title: `Normalize row ${col + 1}`,
        description: 'Scale the pivot row so the pivot becomes 1.',
        matrices: [{ label: 'Normalized pivot row', data: cloneMatrix(augmented) }],
      })
    }

    for (let row = 0; row < n; row += 1) {
      if (row === col) continue
      const factor = augmented[row][col]
      if (Math.abs(factor) < 1e-10) continue
      for (let j = 0; j < 2 * n; j += 1) {
        augmented[row][j] -= factor * augmented[col][j]
      }
      steps.push({
        title: `Eliminate column ${col + 1} for row ${row + 1}`,
        description: `Clear the entry in row ${row + 1}, column ${col + 1}.`,
        matrices: [{ label: 'Column cleared', data: cloneMatrix(augmented) }],
      })
    }
  }

  const inverseMatrix = augmented.map((row) => row.slice(n))
  steps.push({
    title: 'Extract inverse matrix',
    description: 'The right half of the augmented matrix now contains A⁻¹.',
    matrices: [{ label: 'A⁻¹', data: cloneMatrix(inverseMatrix) }],
  })

  return { value: inverseMatrix, steps }
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
  const [visualizationSteps, setVisualizationSteps] = useState<VisualizationStep[] | null>(null)
  const [showVisualization, setShowVisualization] = useState(false)

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
      let nextResult: ResultState
      let nextSteps: VisualizationStep[] = []

      switch (operation) {
        case 'add': {
          const data = addMatrices(numericA, numericB)
          nextResult = { type: 'matrix', label: 'A + B', data }
          const sample =
            numericA[0]?.[0] !== undefined && numericB[0]?.[0] !== undefined
              ? `Example: ${formatNumber(numericA[0][0])} + ${formatNumber(numericB[0][0])} = ${formatNumber(
                  data[0][0],
                )}.`
              : undefined
          nextSteps = [
            {
              title: 'Align matrices',
              description: `Matrix A and Matrix B both have dimensions ${rowsA} × ${colsA}.`,
              matrices: [
                { label: 'Matrix A', data: numericA },
                { label: 'Matrix B', data: numericB },
              ],
            },
            {
              title: 'Add element by element',
              description:
                sample ?? 'Add each pair of corresponding entries aᵢⱼ + bᵢⱼ to build the result.',
            },
            {
              title: 'Resulting matrix',
              matrices: [{ label: 'A + B', data }],
            },
          ]
          break
        }
        case 'subtract': {
          const data = subtractMatrices(numericA, numericB)
          nextResult = { type: 'matrix', label: 'A - B', data }
          const sample =
            numericA[0]?.[0] !== undefined && numericB[0]?.[0] !== undefined
              ? `Example: ${formatNumber(numericA[0][0])} - ${formatNumber(numericB[0][0])} = ${formatNumber(
                  data[0][0],
                )}.`
              : undefined
          nextSteps = [
            {
              title: 'Align matrices',
              description: `Matrix A and Matrix B both have dimensions ${rowsA} × ${colsA}.`,
              matrices: [
                { label: 'Matrix A', data: numericA },
                { label: 'Matrix B', data: numericB },
              ],
            },
            {
              title: 'Subtract element by element',
              description:
                sample ?? 'Subtract each pair of corresponding entries aᵢⱼ - bᵢⱼ to build the result.',
            },
            {
              title: 'Resulting matrix',
              matrices: [{ label: 'A - B', data }],
            },
          ]
          break
        }
        case 'hadamard': {
          const data = hadamardProduct(numericA, numericB)
          nextResult = { type: 'matrix', label: 'A ⊙ B', data }
          const sample =
            numericA[0]?.[0] !== undefined && numericB[0]?.[0] !== undefined
              ? `Example: ${formatNumber(numericA[0][0])} × ${formatNumber(numericB[0][0])} = ${formatNumber(
                  data[0][0],
                )}.`
              : undefined
          nextSteps = [
            {
              title: 'Align matrices',
              description: `Matrix A and Matrix B both have dimensions ${rowsA} × ${colsA}.`,
              matrices: [
                { label: 'Matrix A', data: numericA },
                { label: 'Matrix B', data: numericB },
              ],
            },
            {
              title: 'Multiply element by element',
              description:
                sample ?? 'Multiply each pair of corresponding entries aᵢⱼ · bᵢⱼ to build the result.',
            },
            {
              title: 'Resulting matrix',
              matrices: [{ label: 'A ⊙ B', data }],
            },
          ]
          break
        }
        case 'multiply': {
          const data = multiplyMatrices(numericA, numericB)
          nextResult = { type: 'matrix', label: 'A × B', data }
          const shared = colsA
          const exampleContributions = Array.from({ length: shared }, (_, index) =>
            `${formatNumber(numericA[0]?.[index] ?? 0)} × ${formatNumber(numericB[index]?.[0] ?? 0)}`,
          )
          const exampleValue = formatNumber(data[0]?.[0] ?? 0)
          nextSteps = [
            {
              title: 'Confirm compatibility',
              description: `Matrix A is ${rowsA} × ${colsA} and Matrix B is ${rowsB} × ${colsB}. Shared inner dimension ${colsA} enables multiplication.`,
              matrices: [
                { label: 'Matrix A', data: numericA },
                { label: 'Matrix B', data: numericB },
              ],
            },
            {
              title: 'Compute dot products',
              description:
                shared > 0
                  ? `Each entry cᵢⱼ = Σₖ aᵢₖ · bₖⱼ. Example for c₁₁: ${exampleContributions.join(
                      ' + ',
                    )} = ${exampleValue}.`
                  : 'Each entry cᵢⱼ = Σₖ aᵢₖ · bₖⱼ.',
            },
            {
              title: 'Resulting matrix',
              matrices: [{ label: 'A × B', data }],
            },
          ]
          break
        }
        case 'detA': {
          const { value, steps } = calculateDeterminantWithSteps(numericA)
          nextResult = { type: 'scalar', label: 'det(A)', value }
          nextSteps = steps
          break
        }
        case 'detB': {
          const { value, steps } = calculateDeterminantWithSteps(numericB)
          nextResult = { type: 'scalar', label: 'det(B)', value }
          nextSteps = steps
          break
        }
        case 'invA': {
          const { value, steps } = calculateInverseWithSteps(numericA)
          nextResult = { type: 'matrix', label: 'A⁻¹', data: value }
          nextSteps = steps
          break
        }
        case 'invB': {
          const { value, steps } = calculateInverseWithSteps(numericB)
          nextResult = { type: 'matrix', label: 'B⁻¹', data: value }
          nextSteps = steps
          break
        }
        case 'transA': {
          const data = transpose(numericA)
          nextResult = { type: 'matrix', label: 'Aᵀ', data }
          const example =
            numericA[0]?.[1] !== undefined
              ? `Entry at (1, 2) becomes entry (2, 1): ${formatNumber(numericA[0][1])} → ${formatNumber(
                  data[1]?.[0] ?? 0,
                )}.`
              : 'Swap each entry aᵢⱼ to position aⱼᵢ.'
          nextSteps = [
            {
              title: 'Original matrix',
              matrices: [{ label: 'Matrix A', data: numericA }],
            },
            {
              title: 'Swap rows and columns',
              description: example,
            },
            {
              title: 'Transposed matrix',
              matrices: [{ label: 'Aᵀ', data }],
            },
          ]
          break
        }
        case 'transB': {
          const data = transpose(numericB)
          nextResult = { type: 'matrix', label: 'Bᵀ', data }
          const example =
            numericB[0]?.[1] !== undefined
              ? `Entry at (1, 2) becomes entry (2, 1): ${formatNumber(numericB[0][1])} → ${formatNumber(
                  data[1]?.[0] ?? 0,
                )}.`
              : 'Swap each entry bᵢⱼ to position bⱼᵢ.'
          nextSteps = [
            {
              title: 'Original matrix',
              matrices: [{ label: 'Matrix B', data: numericB }],
            },
            {
              title: 'Swap rows and columns',
              description: example,
            },
            {
              title: 'Transposed matrix',
              matrices: [{ label: 'Bᵀ', data }],
            },
          ]
          break
        }
        default: {
          nextResult = { type: 'message', label: 'Notice', message: 'Select a valid operation.' }
          nextSteps = []
        }
      }

      setResult(nextResult)
      setVisualizationSteps(nextSteps.length > 0 ? nextSteps : null)
      setShowVisualization(false)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'An unexpected error occurred.'
      setResult({ type: 'message', label: 'Calculation error', message })
      setVisualizationSteps(null)
      setShowVisualization(false)
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
              <div className="result-tools">
                <span className="result-chip">Output</span>
                {visualizationSteps?.length ? (
                  <button
                    type="button"
                    className="visualize-button"
                    onClick={() => setShowVisualization((prev) => !prev)}
                    aria-pressed={showVisualization}
                  >
                    {showVisualization ? 'Hide visualization' : 'Visualize'}
                  </button>
                ) : null}
              </div>
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
      {showVisualization && visualizationSteps?.length ? (
        <section className="visualization-section">
          <div className="visualization-card">
            <div className="visualization-header">
              <h3>Computation breakdown</h3>
              <span className="result-chip">Steps</span>
            </div>
            <ol className="visualization-steps">
              {visualizationSteps.map((step, index) => (
                <li key={`${step.title}-${index}`} className="visualization-step">
                  <div className="visualization-step-header">
                    <span className="visualization-step-index">Step {index + 1}</span>
                    <h4>{step.title}</h4>
                  </div>
                  {step.description ? <p>{step.description}</p> : null}
                  {step.matrices ? (
                    <div className="visualization-matrices">
                      {step.matrices.map((matrix, matrixIndex) => (
                        <div key={`${matrix.label}-${matrixIndex}`} className="visualization-matrix">
                          <span className="visualization-matrix-label">{matrix.label}</span>
                          <div
                            className="matrix-grid visualization-grid"
                            style={{
                              gridTemplateColumns: `repeat(${matrix.data[0]?.length ?? 0}, minmax(2.8rem, 1fr))`,
                            }}
                          >
                            {matrix.data.map((row, rowIndex) =>
                              row.map((value, colIndex) => (
                                <div key={`${rowIndex}-${colIndex}`} className="result-cell visualization-cell">
                                  {formatNumber(value)}
                                </div>
                              )),
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : null}
                  {step.scalar ? (
                    <div className="visualization-scalar">
                      <span>{step.scalar.label}</span>
                      <strong>{formatNumber(step.scalar.value)}</strong>
                    </div>
                  ) : null}
                </li>
              ))}
            </ol>
          </div>
        </section>
      ) : null}
    </div>
  )
}

export default App
