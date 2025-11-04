import { useEffect, useMemo, useState } from 'react'
import './App.css'

type Rank = 1 | 2 | 3

type VectorInputState = { rank: 1; shape: [number]; data: string[] }
type MatrixInputState = { rank: 2; shape: [number, number]; data: string[][] }
type Tensor3InputState = { rank: 3; shape: [number, number, number]; data: string[][][] }

type TensorInputState = VectorInputState | MatrixInputState | Tensor3InputState

type NumericVectorState = { rank: 1; shape: [number]; data: number[] }
type NumericMatrixState = { rank: 2; shape: [number, number]; data: number[][] }
type NumericTensor3State = { rank: 3; shape: [number, number, number]; data: number[][][] }

type NumericTensorState = NumericVectorState | NumericMatrixState | NumericTensor3State

type NumericMatrix = number[][]

type ResultState =
  | { type: 'vector'; label: string; data: number[] }
  | { type: 'matrix'; label: string; data: NumericMatrix }
  | { type: 'tensor3'; label: string; data: number[][][] }
  | { type: 'scalar'; label: string; value: number }
  | { type: 'message'; label: string; message: string }

type VisualizationScalar = {
  label: string
  value: number
}

type VisualizationEntity =
  | { type: 'vector'; label: string; data: number[] }
  | { type: 'matrix'; label: string; data: NumericMatrix }
  | { type: 'tensor3'; label: string; data: number[][][] }

type VisualizationStep = {
  title: string
  description?: string
  entities?: VisualizationEntity[]
  scalar?: VisualizationScalar
}

type Operation =
  | 'add'
  | 'subtract'
  | 'hadamard'
  | 'multiply'
  | 'detA'
  | 'detB'
  | 'invA'
  | 'invB'
  | 'transA'
  | 'transB'
  | 'dot'
  | 'outer'
  | 'tensorVector'
  | 'tensorMatMulRight'
  | 'tensorMatMulLeft'

type OperationOutcome = {
  result: ResultState
  steps: VisualizationStep[]
}

type OperationConfig = {
  value: Operation
  label: string
  description: string
  guard: (a: TensorInputState, b: TensorInputState) => boolean
  compute: (a: NumericTensorState, b: NumericTensorState) => OperationOutcome
}

const DIMENSION_OPTIONS = Array.from({ length: 6 }, (_, index) => index + 1)

const DEFAULT_SHAPES: Record<Rank, number[]> = {
  1: [3],
  2: [2, 2],
  3: [2, 2, 2],
}

function normalizeShape(rank: Rank, shape?: number[]): number[] {
  const defaults = DEFAULT_SHAPES[rank]
  return Array.from({ length: rank }, (_, index) => shape?.[index] ?? defaults[index])
}

function createVector(length: number, previous?: string[]): string[] {
  return Array.from({ length }, (_, index) => previous?.[index] ?? '0')
}

function createMatrix(rows: number, cols: number, previous?: string[][]): string[][] {
  return Array.from({ length: rows }, (_, rowIndex) =>
    Array.from({ length: cols }, (_, colIndex) => previous?.[rowIndex]?.[colIndex] ?? '0'),
  )
}

function createTensor3(
  depth: number,
  rows: number,
  cols: number,
  previous?: string[][][],
): string[][][] {
  return Array.from({ length: depth }, (_, depthIndex) =>
    createMatrix(rows, cols, previous?.[depthIndex]),
  )
}

function generateTensorState(
  rank: Rank,
  shape?: number[],
  previous?: TensorInputState,
): TensorInputState {
  const normalized = normalizeShape(rank, shape)
  if (rank === 1) {
    const length = normalized[0]
    const prevData = previous?.rank === 1 ? previous.data : undefined
    return { rank: 1, shape: [length], data: createVector(length, prevData) }
  }
  if (rank === 2) {
    const [rows, cols] = normalized as [number, number]
    const prevData = previous?.rank === 2 ? previous.data : undefined
    return { rank: 2, shape: [rows, cols], data: createMatrix(rows, cols, prevData) }
  }
  const [depth, rows, cols] = normalized as [number, number, number]
  const prevData = previous?.rank === 3 ? previous.data : undefined
  return { rank: 3, shape: [depth, rows, cols], data: createTensor3(depth, rows, cols, prevData) }
}

function updateTensorCell(state: TensorInputState, indices: number[], value: string): TensorInputState {
  if (state.rank === 1) {
    const [index] = indices
    const nextData = state.data.slice()
    nextData[index] = value
    return { ...state, data: nextData }
  }
  if (state.rank === 2) {
    const [row, col] = indices
    const nextData = state.data.map((line) => [...line])
    nextData[row][col] = value
    return { ...state, data: nextData }
  }
  const [depth, row, col] = indices
  const nextData = state.data.map((slice) => slice.map((line) => [...line]))
  nextData[depth][row][col] = value
  return { ...state, data: nextData }
}

function parseCell(value: string): number {
  if (value.trim() === '') return 0
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : 0
}

function parseTensor(state: TensorInputState): NumericTensorState {
  if (state.rank === 1) {
    return { rank: 1, shape: state.shape, data: state.data.map(parseCell) }
  }
  if (state.rank === 2) {
    return {
      rank: 2,
      shape: state.shape,
      data: state.data.map((row) => row.map(parseCell)),
    }
  }
  return {
    rank: 3,
    shape: state.shape,
    data: state.data.map((slice) => slice.map((row) => row.map(parseCell))),
  }
}

function formatShape(shape: number[]): string {
  return shape.length === 1 ? `${shape[0]}` : shape.join(' × ')
}

function formatNumber(value: number): string {
  if (Math.abs(value) < 1e-10) return '0'
  const rounded = Number(value.toFixed(6))
  return Math.abs(rounded) < 1e-10 ? '0' : rounded.toString()
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
      entities: [{ type: 'matrix', label: 'Initial matrix', data: cloneMatrix(working) }],
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
        entities: [{ type: 'matrix', label: 'After row swap', data: cloneMatrix(working) }],
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
      entities: [{ type: 'matrix', label: 'Upper-triangular form', data: cloneMatrix(working) }],
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
      entities: [{ type: 'matrix', label: '[A | I]', data: cloneMatrix(augmented) }],
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
        entities: [{ type: 'matrix', label: 'After row swap', data: cloneMatrix(augmented) }],
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
        entities: [{ type: 'matrix', label: 'Normalized pivot row', data: cloneMatrix(augmented) }],
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
        entities: [{ type: 'matrix', label: 'Column cleared', data: cloneMatrix(augmented) }],
      })
    }
  }

  const inverseMatrix = augmented.map((row) => row.slice(n))
  steps.push({
    title: 'Extract inverse matrix',
    description: 'The right half of the augmented matrix now contains A⁻¹.',
    entities: [{ type: 'matrix', label: 'A⁻¹', data: cloneMatrix(inverseMatrix) }],
  })

  return { value: inverseMatrix, steps }
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

function addTensors(a: NumericTensorState, b: NumericTensorState): NumericTensorState {
  if (a.rank !== b.rank) {
    throw new Error('Addition requires tensors of the same rank and shape.')
  }
  if (a.rank === 1) {
    const vectorB = b as NumericVectorState
    const data = a.data.map((value, index) => value + vectorB.data[index])
    return { rank: 1, shape: a.shape, data }
  }
  if (a.rank === 2) {
    const matrixB = b as NumericMatrixState
    const data = a.data.map((row, i) => row.map((value, j) => value + matrixB.data[i][j]))
    return { rank: 2, shape: a.shape, data }
  }
  const tensorB = b as NumericTensor3State
  const data = a.data.map((slice, i) =>
    slice.map((row, j) => row.map((value, k) => value + tensorB.data[i][j][k])),
  )
  return { rank: 3, shape: a.shape, data }
}

function subtractTensors(a: NumericTensorState, b: NumericTensorState): NumericTensorState {
  if (a.rank !== b.rank) {
    throw new Error('Subtraction requires tensors of the same rank and shape.')
  }
  if (a.rank === 1) {
    const vectorB = b as NumericVectorState
    const data = a.data.map((value, index) => value - vectorB.data[index])
    return { rank: 1, shape: a.shape, data }
  }
  if (a.rank === 2) {
    const matrixB = b as NumericMatrixState
    const data = a.data.map((row, i) => row.map((value, j) => value - matrixB.data[i][j]))
    return { rank: 2, shape: a.shape, data }
  }
  const tensorB = b as NumericTensor3State
  const data = a.data.map((slice, i) =>
    slice.map((row, j) => row.map((value, k) => value - tensorB.data[i][j][k])),
  )
  return { rank: 3, shape: a.shape, data }
}

function hadamardTensors(a: NumericTensorState, b: NumericTensorState): NumericTensorState {
  if (a.rank !== b.rank) {
    throw new Error('Hadamard product requires tensors of the same rank and shape.')
  }
  if (a.rank === 1) {
    const vectorB = b as NumericVectorState
    const data = a.data.map((value, index) => value * vectorB.data[index])
    return { rank: 1, shape: a.shape, data }
  }
  if (a.rank === 2) {
    const matrixB = b as NumericMatrixState
    const data = a.data.map((row, i) => row.map((value, j) => value * matrixB.data[i][j]))
    return { rank: 2, shape: a.shape, data }
  }
  const tensorB = b as NumericTensor3State
  const data = a.data.map((slice, i) =>
    slice.map((row, j) => row.map((value, k) => value * tensorB.data[i][j][k])),
  )
  return { rank: 3, shape: a.shape, data }
}

function tensorsShareShape(a: TensorInputState, b: TensorInputState): boolean {
  if (a.rank !== b.rank) return false
  if (a.shape.length !== b.shape.length) return false
  return a.shape.every((value, index) => value === b.shape[index])
}

function tensorVectorContraction(
  a: NumericTensor3State,
  b: NumericVectorState,
): NumericMatrixState {
  const [depth, rows, cols] = a.shape
  if (b.shape[0] !== cols) {
    throw new Error('Tensor contraction requires the vector length to match the tensor axis.')
  }

  const data = Array.from({ length: depth }, (_, depthIndex) =>
    Array.from({ length: rows }, (_, rowIndex) => {
      let sum = 0
      for (let colIndex = 0; colIndex < cols; colIndex += 1) {
        sum += a.data[depthIndex][rowIndex][colIndex] * b.data[colIndex]
      }
      return sum
    }),
  )

  return { rank: 2, shape: [depth, rows], data }
}

function tensorMatrixRight(
  a: NumericTensor3State,
  b: NumericMatrixState,
): NumericTensor3State {
  const [depth, rows, cols] = a.shape
  const [bRows, bCols] = b.shape
  if (cols !== bRows) {
    throw new Error('Shared dimension mismatch for tensor × matrix multiplication.')
  }

  const data = a.data.map((slice) => multiplyMatrices(slice, b.data))
  return { rank: 3, shape: [depth, rows, bCols], data }
}

function matrixTensorLeft(
  a: NumericMatrixState,
  b: NumericTensor3State,
): NumericTensor3State {
  const [rowsA, colsA] = a.shape
  const [depthB, rowsB, colsB] = b.shape
  if (colsA !== rowsB) {
    throw new Error('Shared dimension mismatch for matrix × tensor multiplication.')
  }

  const data = b.data.map((slice) => multiplyMatrices(a.data, slice))
  return { rank: 3, shape: [depthB, rowsA, colsB], data }
}

function dotProduct(
  a: NumericVectorState,
  b: NumericVectorState,
): { value: number; steps: VisualizationStep[] } {
  const length = a.shape[0]
  let accumulator = 0
  const partials: number[] = []
  for (let index = 0; index < length; index += 1) {
    const product = a.data[index] * b.data[index]
    partials.push(product)
    accumulator += product
  }

  const steps: VisualizationStep[] = [
    {
      title: 'Pair elements',
      description: `Multiply each pair of entries across ${length} positions.`,
      entities: [
        { type: 'vector', label: 'Vector A', data: a.data },
        { type: 'vector', label: 'Vector B', data: b.data },
      ],
    },
    {
      title: 'Accumulate products',
      description: `Sum the products ${partials.map((value) => formatNumber(value)).join(' + ')}.`,
      scalar: { label: 'Dot product', value: accumulator },
    },
  ]

  return { value: accumulator, steps }
}

function outerProduct(
  a: NumericVectorState,
  b: NumericVectorState,
): { data: NumericMatrix; steps: VisualizationStep[] } {
  const data = Array.from({ length: a.shape[0] }, (_, i) =>
    Array.from({ length: b.shape[0] }, (_, j) => a.data[i] * b.data[j]),
  )

  const steps: VisualizationStep[] = [
    {
      title: 'Arrange vectors',
      description: 'Treat Vector A as rows and Vector B as columns.',
      entities: [
        { type: 'vector', label: 'Vector A', data: a.data },
        { type: 'vector', label: 'Vector B', data: b.data },
      ],
    },
    {
      title: 'Multiply pairwise',
      description: 'Each cell is aᵢ × bⱼ forming the rank-1 matrix.',
      entities: [{ type: 'matrix', label: 'Outer product', data }],
    },
  ]

  return { data, steps }
}

function createVisualizationEntity(label: string, tensor: NumericTensorState): VisualizationEntity {
  if (tensor.rank === 1) {
    return { type: 'vector', label, data: tensor.data }
  }
  if (tensor.rank === 2) {
    return { type: 'matrix', label, data: tensor.data }
  }
  return { type: 'tensor3', label, data: tensor.data }
}

function resultFromTensor(label: string, tensor: NumericTensorState): ResultState {
  if (tensor.rank === 1) {
    return { type: 'vector', label, data: tensor.data }
  }
  if (tensor.rank === 2) {
    return { type: 'matrix', label, data: tensor.data }
  }
  return { type: 'tensor3', label, data: tensor.data }
}

function samplePath(rank: Rank): string {
  if (rank === 1) return '[0]'
  if (rank === 2) return '[0, 0]'
  return '[0, 0, 0]'
}

function sampleValue(tensor: NumericTensorState): number {
  if (tensor.rank === 1) return tensor.data[0] ?? 0
  if (tensor.rank === 2) return tensor.data[0]?.[0] ?? 0
  return tensor.data[0]?.[0]?.[0] ?? 0
}

function tensorAdditionSteps(
  a: NumericTensorState,
  b: NumericTensorState,
  result: NumericTensorState,
  operation: '+' | '-' | '⊙',
): VisualizationStep[] {
  const shapeText = formatShape(a.shape)
  const path = samplePath(a.rank)
  const aSample = sampleValue(a)
  const bSample = sampleValue(b)
  const rSample = sampleValue(result)

  return [
    {
      title: 'Confirm shape compatibility',
      description: `Tensor A and Tensor B both have shape ${shapeText}.`,
      entities: [createVisualizationEntity('Tensor A', a), createVisualizationEntity('Tensor B', b)],
    },
    {
      title: 'Operate element-wise',
      description: `Example at index ${path}: ${formatNumber(aSample)} ${operation} ${formatNumber(
        bSample,
      )} = ${formatNumber(rSample)}.`,
    },
    {
      title: 'Resulting tensor',
      entities: [createVisualizationEntity('Result', result)],
    },
  ]
}
const OPERATION_CONFIG: OperationConfig[] = [
  {
    value: 'add',
    label: 'A + B',
    description: 'Element-wise addition for tensors sharing the same shape.',
    guard: (a, b) => tensorsShareShape(a, b),
    compute: (a, b) => {
      const tensor = addTensors(a, b)
      const steps = tensorAdditionSteps(a, b, tensor, '+')
      return { result: resultFromTensor('A + B', tensor), steps }
    },
  },
  {
    value: 'subtract',
    label: 'A - B',
    description: 'Element-wise subtraction of Tensor B from Tensor A.',
    guard: (a, b) => tensorsShareShape(a, b),
    compute: (a, b) => {
      const tensor = subtractTensors(a, b)
      const steps = tensorAdditionSteps(a, b, tensor, '-')
      return { result: resultFromTensor('A - B', tensor), steps }
    },
  },
  {
    value: 'hadamard',
    label: 'A ⊙ B (Hadamard)',
    description: 'Element-wise multiplication, useful for gating and feature scaling.',
    guard: (a, b) => tensorsShareShape(a, b),
    compute: (a, b) => {
      const tensor = hadamardTensors(a, b)
      const steps = tensorAdditionSteps(a, b, tensor, '⊙')
      return { result: resultFromTensor('A ⊙ B', tensor), steps }
    },
  },
  {
    value: 'multiply',
    label: 'A × B',
    description: 'Matrix multiplication across the inner dimension.',
    guard: (a, b) => a.rank === 2 && b.rank === 2 && a.shape[1] === b.shape[0],
    compute: (a, b) => {
      const matrixA = a as NumericMatrixState
      const matrixB = b as NumericMatrixState
      const product = multiplyMatrices(matrixA.data, matrixB.data)
      const steps: VisualizationStep[] = [
        {
          title: 'Check inner dimensions',
          description: `A is ${formatShape(matrixA.shape)} and B is ${formatShape(matrixB.shape)} so ${matrixA.shape[1]} = ${matrixB.shape[0]}.`,
        },
        {
          title: 'Multiply rows by columns',
          description: 'Each entry cᵢⱼ is the dot product of row i of A with column j of B.',
          entities: [
            createVisualizationEntity('Matrix A', matrixA),
            createVisualizationEntity('Matrix B', matrixB),
          ],
        },
        {
          title: 'Resulting matrix',
          entities: [{ type: 'matrix', label: 'A × B', data: product }],
        },
      ]
      return { result: { type: 'matrix', label: 'A × B', data: product }, steps }
    },
  },
  {
    value: 'detA',
    label: 'det(A)',
    description: 'Determinant of tensor A when it is a square matrix.',
    guard: (a) => a.rank === 2 && a.shape[0] === a.shape[1],
    compute: (a) => {
      const matrixA = a as NumericMatrixState
      const { value, steps } = calculateDeterminantWithSteps(matrixA.data)
      return { result: { type: 'scalar', label: 'det(A)', value }, steps }
    },
  },
  {
    value: 'detB',
    label: 'det(B)',
    description: 'Determinant of tensor B when it is a square matrix.',
    guard: (_, b) => b.rank === 2 && b.shape[0] === b.shape[1],
    compute: (_, b) => {
      const matrixB = b as NumericMatrixState
      const { value, steps } = calculateDeterminantWithSteps(matrixB.data)
      return { result: { type: 'scalar', label: 'det(B)', value }, steps }
    },
  },
  {
    value: 'invA',
    label: 'A⁻¹',
    description: 'Inverse of tensor A when it is a square, non-singular matrix.',
    guard: (a) => a.rank === 2 && a.shape[0] === a.shape[1],
    compute: (a) => {
      const matrixA = a as NumericMatrixState
      const { value, steps } = calculateInverseWithSteps(matrixA.data)
      return { result: { type: 'matrix', label: 'A⁻¹', data: value }, steps }
    },
  },
  {
    value: 'invB',
    label: 'B⁻¹',
    description: 'Inverse of tensor B when it is a square, non-singular matrix.',
    guard: (_, b) => b.rank === 2 && b.shape[0] === b.shape[1],
    compute: (_, b) => {
      const matrixB = b as NumericMatrixState
      const { value, steps } = calculateInverseWithSteps(matrixB.data)
      return { result: { type: 'matrix', label: 'B⁻¹', data: value }, steps }
    },
  },
  {
    value: 'transA',
    label: 'Aᵀ',
    description: 'Transpose tensor A when it is a matrix.',
    guard: (a) => a.rank === 2,
    compute: (a) => {
      const matrixA = a as NumericMatrixState
      const rows = matrixA.data.length
      const cols = matrixA.data[0]?.length ?? 0
      const transpose = Array.from({ length: cols }, (_, col) =>
        Array.from({ length: rows }, (_, row) => matrixA.data[row][col]),
      )
      const steps: VisualizationStep[] = [
        {
          title: 'Swap rows and columns',
          description: 'The (i, j) entry becomes (j, i) in the transpose.',
          entities: [
            createVisualizationEntity('Matrix A', matrixA),
            { type: 'matrix', label: 'Aᵀ', data: transpose },
          ],
        },
      ]
      return { result: { type: 'matrix', label: 'Aᵀ', data: transpose }, steps }
    },
  },
  {
    value: 'transB',
    label: 'Bᵀ',
    description: 'Transpose tensor B when it is a matrix.',
    guard: (_, b) => b.rank === 2,
    compute: (_, b) => {
      const matrixB = b as NumericMatrixState
      const rows = matrixB.data.length
      const cols = matrixB.data[0]?.length ?? 0
      const transpose = Array.from({ length: cols }, (_, col) =>
        Array.from({ length: rows }, (_, row) => matrixB.data[row][col]),
      )
      const steps: VisualizationStep[] = [
        {
          title: 'Swap rows and columns',
          description: 'The (i, j) entry becomes (j, i) in the transpose.',
          entities: [
            createVisualizationEntity('Matrix B', matrixB),
            { type: 'matrix', label: 'Bᵀ', data: transpose },
          ],
        },
      ]
      return { result: { type: 'matrix', label: 'Bᵀ', data: transpose }, steps }
    },
  },
  {
    value: 'dot',
    label: '⟨A, B⟩ (Dot)',
    description: 'Dot product between vectors A and B.',
    guard: (a, b) => a.rank === 1 && b.rank === 1 && a.shape[0] === b.shape[0],
    compute: (a, b) => {
      const { value, steps } = dotProduct(a as NumericVectorState, b as NumericVectorState)
      return { result: { type: 'scalar', label: '⟨A, B⟩', value }, steps }
    },
  },
  {
    value: 'outer',
    label: 'A ⊗ B (Outer)',
    description: 'Outer product between vectors A and B forming a rank-1 matrix.',
    guard: (a, b) => a.rank === 1 && b.rank === 1,
    compute: (a, b) => {
      const { data, steps } = outerProduct(a as NumericVectorState, b as NumericVectorState)
      return { result: { type: 'matrix', label: 'A ⊗ B', data }, steps }
    },
  },
  {
    value: 'tensorVector',
    label: 'Tensor A × Vector B',
    description: 'Contract Tensor A along its last axis with Vector B.',
    guard: (a, b) => a.rank === 3 && b.rank === 1 && a.shape[2] === b.shape[0],
    compute: (a, b) => {
      const tensorA = a as NumericTensor3State
      const vectorB = b as NumericVectorState
      const contraction = tensorVectorContraction(tensorA, vectorB)
      const steps: VisualizationStep[] = [
        {
          title: 'Match contraction axis',
          description: `Tensor A shape ${formatShape(tensorA.shape)} contracts with vector length ${vectorB.shape[0]}.`,
          entities: [
            createVisualizationEntity('Tensor A', tensorA),
            createVisualizationEntity('Vector B', vectorB),
          ],
        },
        {
          title: 'Compute slice dot products',
          description: 'Each row across the last axis performs a dot product with Vector B.',
        },
        {
          title: 'Resulting matrix',
          entities: [{ type: 'matrix', label: 'Contraction result', data: contraction.data }],
        },
      ]
      return { result: { type: 'matrix', label: 'Tensor A × Vector B', data: contraction.data }, steps }
    },
  },
  {
    value: 'tensorMatMulRight',
    label: 'Tensor A × Matrix B',
    description: 'Multiply each frontal slice of Tensor A by Matrix B on the right.',
    guard: (a, b) => a.rank === 3 && b.rank === 2 && a.shape[2] === b.shape[0],
    compute: (a, b) => {
      const tensorA = a as NumericTensor3State
      const matrixB = b as NumericMatrixState
      const product = tensorMatrixRight(tensorA, matrixB)
      const steps: VisualizationStep[] = [
        {
          title: 'Validate shared axis',
          description: `Tensor A shape ${formatShape(tensorA.shape)} shares ${tensorA.shape[2]} with Matrix B shape ${formatShape(matrixB.shape)}.`,
        },
        {
          title: 'Multiply per slice',
          description: 'For each depth slice, perform standard matrix multiplication with B.',
          entities: [
            createVisualizationEntity('Tensor A', tensorA),
            createVisualizationEntity('Matrix B', matrixB),
          ],
        },
        {
          title: 'Resulting tensor',
          entities: [{ type: 'tensor3', label: 'Product tensor', data: product.data }],
        },
      ]
      return { result: { type: 'tensor3', label: 'Tensor A × Matrix B', data: product.data }, steps }
    },
  },
  {
    value: 'tensorMatMulLeft',
    label: 'Matrix A × Tensor B',
    description: 'Multiply Matrix A with each frontal slice of Tensor B on the left.',
    guard: (a, b) => a.rank === 2 && b.rank === 3 && a.shape[1] === b.shape[1],
    compute: (a, b) => {
      const matrixA = a as NumericMatrixState
      const tensorB = b as NumericTensor3State
      const product = matrixTensorLeft(matrixA, tensorB)
      const steps: VisualizationStep[] = [
        {
          title: 'Validate shared axis',
          description: `Matrix A shape ${formatShape(matrixA.shape)} shares ${matrixA.shape[1]} with Tensor B shape ${formatShape(tensorB.shape)}.`,
        },
        {
          title: 'Multiply per slice',
          description: 'For each tensor slice, multiply on the left by Matrix A.',
          entities: [
            createVisualizationEntity('Matrix A', matrixA),
            createVisualizationEntity('Tensor B', tensorB),
          ],
        },
        {
          title: 'Resulting tensor',
          entities: [{ type: 'tensor3', label: 'Product tensor', data: product.data }],
        },
      ]
      return { result: { type: 'tensor3', label: 'Matrix A × Tensor B', data: product.data }, steps }
    },
  },
];
type DimensionLabelSet = {
  rank: Rank
  labels: string[]
}

const DIMENSION_LABELS: DimensionLabelSet[] = [
  { rank: 1, labels: ['Length'] },
  { rank: 2, labels: ['Rows', 'Cols'] },
  { rank: 3, labels: ['Depth', 'Rows', 'Cols'] },
]

type TensorEditorProps = {
  name: 'A' | 'B'
  state: TensorInputState
  onRankChange: (rank: Rank) => void
  onShapeChange: (axis: number, value: number) => void
  onCellChange: (indices: number[], value: string) => void
}

function TensorEditor({ name, state, onRankChange, onShapeChange, onCellChange }: TensorEditorProps) {
  const dimensionLabels = DIMENSION_LABELS.find((entry) => entry.rank === state.rank) ?? DIMENSION_LABELS[0]

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
          <h2>{state.rank === 1 ? `Vector ${name}` : state.rank === 2 ? `Matrix ${name}` : `Tensor ${name}`}</h2>
          <p className="matrix-subtitle">Configure rank, dimensions, and values.</p>
        </div>
        <div className="tensor-rank-controls" role="group" aria-label={`Rank selection for ${name}`}>
          {[1, 2, 3].map((rank) => (
            <button
              key={rank}
              type="button"
              className={state.rank === rank ? 'rank-button active' : 'rank-button'}
              onClick={() => onRankChange(rank as Rank)}
              aria-pressed={state.rank === rank}
            >
              Rank {rank}
            </button>
          ))}
        </div>
      </div>
      <div className="dimension-controls">
        {dimensionLabels.labels.map((label, index) => (
          <label key={label} className="dimension-control">
            <span>{label}</span>
            <select
              value={state.shape[index]}
              onChange={(event) => onShapeChange(index, Number(event.target.value))}
            >
              {DIMENSION_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
        ))}
      </div>
      {state.rank === 1 ? (
        <div className="vector-grid" aria-label={`Vector ${name} values`}>
          {state.data.map((value, index) => (
            <input
              key={index}
              type="number"
              inputMode="decimal"
              step="any"
              value={value}
              onChange={(event) => onCellChange([index], event.target.value)}
              aria-label={`Vector ${name} entry ${index + 1}`}
            />
          ))}
        </div>
      ) : null}
      {state.rank === 2 ? (
        <div
          className="matrix-grid"
          style={{ gridTemplateColumns: `repeat(${state.shape[1]}, minmax(3.5rem, 1fr))` }}
          aria-label={`Matrix ${name} grid`}
        >
          {state.data.map((row, rowIndex) =>
            row.map((value, colIndex) => (
              <input
                key={`${rowIndex}-${colIndex}`}
                type="number"
                inputMode="decimal"
                step="any"
                value={value}
                onChange={(event) => onCellChange([rowIndex, colIndex], event.target.value)}
                aria-label={`Matrix ${name} cell ${rowIndex + 1}, ${colIndex + 1}`}
              />
            )),
          )}
        </div>
      ) : null}
      {state.rank === 3 ? (
        <div className="tensor-slices-editor" aria-label={`Tensor ${name} slices`}>
          {state.data.map((slice, depthIndex) => (
            <div key={depthIndex} className="tensor-slice-card">
              <span className="tensor-slice-label">Slice {depthIndex + 1}</span>
              <div
                className="matrix-grid"
                style={{ gridTemplateColumns: `repeat(${state.shape[2]}, minmax(3.5rem, 1fr))` }}
              >
                {slice.map((row, rowIndex) =>
                  row.map((value, colIndex) => (
                    <input
                      key={`${depthIndex}-${rowIndex}-${colIndex}`}
                      type="number"
                      inputMode="decimal"
                      step="any"
                      value={value}
                      onChange={(event) =>
                        onCellChange([depthIndex, rowIndex, colIndex], event.target.value)
                      }
                      aria-label={`Tensor ${name} slice ${depthIndex + 1}, row ${rowIndex + 1}, col ${colIndex + 1}`}
                    />
                  )),
                )}
              </div>
            </div>
          ))}
        </div>
      ) : null}
      <div className="matrix-footnote">Shape {formatShape(state.shape)}</div>
    </div>
  )
}

type DisplayVariant = 'result' | 'visualization'

function VectorDisplay({ data, variant = 'result' }: { data: number[]; variant?: DisplayVariant }) {
  const gridClass =
    variant === 'visualization' ? 'vector-display visualization-vector' : 'vector-display'
  const cellClass = variant === 'visualization' ? 'result-cell visualization-cell' : 'result-cell'
  return (
    <div className={gridClass}>
      {data.map((value, index) => (
        <div key={index} className={cellClass}>
          {formatNumber(value)}
        </div>
      ))}
    </div>
  )
}

function MatrixDisplay({
  data,
  variant = 'result',
}: {
  data: NumericMatrix
  variant?: DisplayVariant
}) {
  if (!data.length) {
    return <p className="result-message">Empty matrix</p>
  }
  const columns = data[0]?.length ?? 0
  const cellWidth = variant === 'visualization' ? '2.8rem' : '3.5rem'
  const gridClass =
    variant === 'visualization' ? 'matrix-grid visualization-grid' : 'matrix-grid result-grid'
  const cellClass = variant === 'visualization' ? 'result-cell visualization-cell' : 'result-cell'
  return (
    <div className={gridClass} style={{ gridTemplateColumns: `repeat(${columns}, minmax(${cellWidth}, 1fr))` }}>
      {data.map((row, rowIndex) =>
        row.map((value, colIndex) => (
          <div key={`${rowIndex}-${colIndex}`} className={cellClass}>
            {formatNumber(value)}
          </div>
        )),
      )}
    </div>
  )
}

function Tensor3Display({
  data,
  variant = 'result',
}: {
  data: number[][][]
  variant?: DisplayVariant
}) {
  if (!data.length) {
    return <p className="result-message">Empty tensor</p>
  }
  const wrapperClass =
    variant === 'visualization' ? 'tensor-slices visualization-tensor' : 'tensor-slices'
  return (
    <div className={wrapperClass}>
      {data.map((slice, index) => (
        <div key={index} className="tensor-slice-card">
          <span className="tensor-slice-label">Slice {index + 1}</span>
          <MatrixDisplay data={slice} variant={variant} />
        </div>
      ))}
    </div>
  )
}

function VisualizationEntityView({ entity }: { entity: VisualizationEntity }) {
  return (
    <div className="visualization-entity">
      <span className="visualization-matrix-label">{entity.label}</span>
      {entity.type === 'vector' ? (
        <VectorDisplay data={entity.data} variant="visualization" />
      ) : entity.type === 'matrix' ? (
        <MatrixDisplay data={entity.data} variant="visualization" />
      ) : (
        <Tensor3Display data={entity.data} variant="visualization" />
      )}
    </div>
  )
}

function ResultBody({ result }: { result: ResultState }) {
  if (result.type === 'vector') {
    return <VectorDisplay data={result.data} />
  }
  if (result.type === 'matrix') {
    return <MatrixDisplay data={result.data} />
  }
  if (result.type === 'tensor3') {
    return <Tensor3Display data={result.data} />
  }
  if (result.type === 'scalar') {
    return <div className="scalar-result">{formatNumber(result.value)}</div>
  }
  if (result.type === 'message') {
    return <p className="result-message">{result.message}</p>
  }
  return null
}

function App() {
  const [tensorA, setTensorA] = useState<TensorInputState>(() => generateTensorState(2))
  const [tensorB, setTensorB] = useState<TensorInputState>(() => generateTensorState(2))
  const [operation, setOperation] = useState<Operation>('add')
  const [result, setResult] = useState<ResultState | null>(null)
  const [visualizationSteps, setVisualizationSteps] = useState<VisualizationStep[] | null>(null)
  const [showVisualization, setShowVisualization] = useState(false)

  const availableOperations = useMemo(
    () => OPERATION_CONFIG.filter((config) => config.guard(tensorA, tensorB)),
    [tensorA, tensorB],
  )

  useEffect(() => {
    if (!availableOperations.some((config) => config.value === operation)) {
      setOperation(availableOperations[0]?.value ?? 'add')
    }
  }, [availableOperations, operation])

  const selectedOperation = availableOperations.find((config) => config.value === operation)

  const handleRankChange = (name: 'A' | 'B') => (rank: Rank) => {
    if (name === 'A') {
      setTensorA((prev) => (prev.rank === rank ? prev : generateTensorState(rank)))
    } else {
      setTensorB((prev) => (prev.rank === rank ? prev : generateTensorState(rank)))
    }
  }

  const handleShapeChange = (name: 'A' | 'B') => (axis: number, value: number) => {
    if (name === 'A') {
      setTensorA((prev) => {
        const nextShape = [...prev.shape] as number[]
        nextShape[axis] = value
        return generateTensorState(prev.rank, nextShape, prev)
      })
    } else {
      setTensorB((prev) => {
        const nextShape = [...prev.shape] as number[]
        nextShape[axis] = value
        return generateTensorState(prev.rank, nextShape, prev)
      })
    }
  }

  const handleCellChange = (name: 'A' | 'B') => (indices: number[], value: string) => {
    if (name === 'A') {
      setTensorA((prev) => updateTensorCell(prev, indices, value))
    } else {
      setTensorB((prev) => updateTensorCell(prev, indices, value))
    }
  }

  const handleCalculate = () => {
    const numericA = parseTensor(tensorA)
    const numericB = parseTensor(tensorB)
    const config = availableOperations.find((entry) => entry.value === operation)
    if (!config) {
      setResult({
        type: 'message',
        label: 'Notice',
        message: 'Select a valid operation for the current tensors.',
      })
      setVisualizationSteps(null)
      setShowVisualization(false)
      return
    }

    try {
      const { result: nextResult, steps } = config.compute(numericA, numericB)
      setResult(nextResult)
      setVisualizationSteps(steps.length ? steps : null)
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
          <h1>Tensor Lab for Machine Learning</h1>
          <p>
            Explore vectors, matrices, and rank-3 tensors with compatibility-aware operations and
            detailed computation breakdowns.
          </p>
        </div>
        <div className="operation-chip">Dark mode · Tensor-ready</div>
      </header>

      <section className="matrix-section">
        <TensorEditor
          name="A"
          state={tensorA}
          onRankChange={handleRankChange('A')}
          onShapeChange={handleShapeChange('A')}
          onCellChange={handleCellChange('A')}
        />
        <TensorEditor
          name="B"
          state={tensorB}
          onRankChange={handleRankChange('B')}
          onShapeChange={handleShapeChange('B')}
          onCellChange={handleCellChange('B')}
        />
      </section>

      <section className="operation-section">
        <div className="operation-panel">
          <div className="operation-heading">
            <h2>Choose a computation</h2>
            <p>Select an operation available for the current tensor configurations.</p>
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
          {selectedOperation ? (
            <p className="operation-description">{selectedOperation.description}</p>
          ) : null}
        </div>
      </section>

      {result ? (
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
            <ResultBody result={result} />
          </div>
        </section>
      ) : null}

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
                  {step.entities ? (
                    <div className="visualization-entities">
                      {step.entities.map((entity, entityIndex) => (
                        <VisualizationEntityView
                          key={`${entity.label}-${entityIndex}`}
                          entity={entity}
                        />
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
  );
};

export default App;
